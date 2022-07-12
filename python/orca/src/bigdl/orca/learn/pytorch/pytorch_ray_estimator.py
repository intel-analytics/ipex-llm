#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import io
import types
import logging
import numbers
import torch
import numpy as np

from bigdl.orca.data.ray_xshards import RayXShards
from bigdl.orca.learn.pytorch.training_operator import TrainingOperator
from bigdl.orca.learn.pytorch.pytorch_ray_worker import PytorchRayWorker
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, update_predict_xshards, \
    process_xshards_of_pandas_dataframe
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca.learn.ray_estimator import Estimator as OrcaRayEstimator
from bigdl.dllib.utils.file_utils import enable_multi_fs_load, enable_multi_fs_save

import ray
from ray.exceptions import RayActorError
from bigdl.dllib.utils.log4Error import *


logger = logging.getLogger(__name__)


def check_for_failure(remote_values):
    """Checks remote values for any that returned and failed.
    :param remote_values: List of object IDs representing functions
            that may fail in the middle of execution. For example, running
            a SGD training loop in multiple parallel actor calls.
    :return Bool for success in executing given remote tasks.
    """
    unfinished = remote_values
    try:
        while len(unfinished) > 0:
            finished, unfinished = ray.wait(unfinished)
            finished = ray.get(finished)
        return True
    except RayActorError as exc:
        logger.exception(str(exc))
    return False


def partition_refs_to_creator(partition_refs):
    def data_creator(config, batch_size):
        from bigdl.orca.data.utils import ray_partitions_get_data_label, index_data, get_size
        from torch.utils.data import Dataset, DataLoader

        class NDArrayDataset(Dataset):
            def __init__(self, x, y):
                self.x = x  # features
                self.y = y  # labels

            def __len__(self):
                return get_size(self.y)

            def __getitem__(self, i):
                return index_data(self.x, i), index_data(self.y, i)

        params = {"batch_size": batch_size, "shuffle": True}
        for arg in ["shuffle", "sampler", "batch_sampler", "num_workers", "collate_fn",
                    "pin_memory", "drop_last", "timeout", "worker_init_fn",
                    "multiprocessing_context"]:
            if arg in config:
                params[arg] = config[arg]
        data, label = ray_partitions_get_data_label(ray.get(partition_refs),
                                                    allow_tuple=False,
                                                    allow_list=False)
        print("Data size on worker: ", len(label))
        dataset = NDArrayDataset(data, label)
        data_loader = DataLoader(dataset, **params)
        return data_loader

    return data_creator


class PyTorchRayEstimator(OrcaRayEstimator):
    def __init__(
            self,
            *,
            model_creator,
            optimizer_creator,
            loss_creator=None,
            metrics=None,
            scheduler_creator=None,
            training_operator_cls=TrainingOperator,
            initialization_hook=None,
            config=None,
            scheduler_step_freq="batch",
            use_tqdm=False,
            backend="ray",
            workers_per_node=1,
            sync_stats=True,
            log_level=logging.INFO):
        if config is not None and "batch_size" in config:
            invalidInputError(False,
                              "Please do not specify batch_size in config. Input batch_size in the"
                              " fit/evaluate/predict function of the estimator instead.")

        # todo remove ray_ctx to run on workers
        ray_ctx = OrcaRayContext.get()
        if not (isinstance(model_creator, types.FunctionType) and
                isinstance(optimizer_creator, types.FunctionType)):  # Torch model is also callable.
            invalidInputError(False,
                              "Must provide a function for both model_creator and"
                              " optimizer_creator")

        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.scheduler_creator = scheduler_creator
        self.training_operator_cls = training_operator_cls
        self.scheduler_step_freq = scheduler_step_freq
        self.use_tqdm = use_tqdm
        self.sync_stats = sync_stats
        self.backend = backend

        if not training_operator_cls and not loss_creator:
            invalidInputError(False,
                              "If a loss_creator is not provided, you must "
                              "provide a custom training operator.")

        self.initialization_hook = initialization_hook
        self.config = {} if config is None else config
        worker_config = self.config.copy()
        params = dict(
            model_creator=self.model_creator,
            optimizer_creator=self.optimizer_creator,
            loss_creator=self.loss_creator,
            scheduler_creator=self.scheduler_creator,
            training_operator_cls=self.training_operator_cls,
            scheduler_step_freq=self.scheduler_step_freq,
            use_tqdm=self.use_tqdm,
            config=worker_config,
            metrics=metrics,
            sync_stats=sync_stats,
            log_level=log_level
        )

        if backend == "ray":
            import torch.distributed as dist
            cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            num_nodes = ray_ctx.num_ray_nodes * workers_per_node
            RemoteRunner = ray.remote(num_cpus=cores_per_node)(PytorchRayWorker)
            self.remote_workers = [
                RemoteRunner.remote(**params) for i in range(num_nodes)
            ]
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            head_worker = self.remote_workers[0]
            driver_ip, driver_tcp_store_port = ray.get(head_worker.get_node_ip_port.remote())

            _ = dist.TCPStore(driver_ip, driver_tcp_store_port, -1, True,
                              dist.constants.default_pg_timeout)

            ray.get([
                worker.setup_torch_distribute.remote(
                    driver_ip, driver_tcp_store_port, i, num_nodes)
                for i, worker in enumerate(self.remote_workers)
            ])

        elif backend == "horovod":
            from bigdl.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
            self.horovod_runner = HorovodRayRunner(ray_ctx,
                                                   worker_cls=PytorchRayWorker,
                                                   worker_param=params,
                                                   workers_per_node=workers_per_node)
            self.remote_workers = self.horovod_runner.remote_workers
            cores_per_node = self.horovod_runner.cores_per_node
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            ray.get([
                worker.setup_horovod.remote()
                for i, worker in enumerate(self.remote_workers)
            ])
        else:
            invalidInputError(False,
                              "Only \"ray\" and \"horovod\" are supported "
                              "values of backend, but got {}".format(backend))
        self.num_workers = len(self.remote_workers)

    def fit(self,
            data,
            epochs=1,
            batch_size=32,
            profile=False,
            reduce_results=True,
            info=None,
            feature_cols=None,
            label_cols=None,
            validation_data=None,
            callbacks=[]):
        """
        Trains a PyTorch model given training data for several epochs.
        Calls `TrainingOperator.train_epoch()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Ray Dataset, a Spark DataFrame or a function
               that takes config and batch_size as argument and returns a PyTorch DataLoader for
               training.
        :param epochs: The number of epochs to train the model. Default is 1.
        :param batch_size: The number of samples per batch for each worker. Default is 32.
               The total batch size would be workers_per_node*num_nodes.
               If your training data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param reduce_results: Boolean. Whether to average all metrics across all workers into
               one dict. If a metric is a non-numerical value (or nested dictionaries), one value
               will be randomly selected among the workers. If False, returns a list of dicts for
               all workers.
               Default is True.
        :param info: An optional dictionary that can be passed to the TrainingOperator for
               train_epoch and train_batch.
        :param feature_cols: feature column names if data is Spark DataFrame or Ray Dataset.
        :param label_cols: label column names if data is Spark DataFrame or Ray Dataset.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param callbacks: A list for all callbacks.

        :return: A list of dictionary of metrics for every training epoch. If reduce_results is
                False, this will return a nested list of metric dictionaries whose length will be
                equal to the total number of workers.
                You can also provide custom metrics by passing in a custom training_operator_cls
                when creating the Estimator.
        """
        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            profile=profile,
            info=info,
            callbacks=callbacks,
        )

        from bigdl.orca.data import SparkXShards
        from ray.data import Dataset

        data, validation_data = maybe_dataframe_to_xshards(data,
                                                           validation_data=validation_data,
                                                           feature_cols=feature_cols,
                                                           label_cols=label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers)

        if isinstance(data, SparkXShards):
            # Should not wrap DistributedSampler on DataLoader for SparkXShards input.
            params["wrap_dataloader"] = False
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data, "fit")
            from bigdl.orca.data.utils import process_spark_xshards
            ray_xshards = process_spark_xshards(data, self.num_workers)

            if validation_data is None:
                def transform_func(worker, partition_refs):
                    data_creator = partition_refs_to_creator(partition_refs)
                    params["data_creator"] = data_creator
                    return worker.train_epochs.remote(**params)

                worker_stats = ray_xshards.reduce_partitions_for_actors(self.remote_workers,
                                                                        transform_func)
            else:
                if self.backend == "horovod":
                    invalidInputError(False,
                                      "Currently, we don't support input validation_data"
                                      " for horovod backend")
                val_ray_xshards = process_spark_xshards(validation_data, self.num_workers)

                def zip_func(worker, this_partition_refs, that_partition_refs):
                    params["data_creator"] = partition_refs_to_creator(this_partition_refs)
                    params["validation_data_creator"] = \
                        partition_refs_to_creator(that_partition_refs)
                    return worker.train_epochs.remote(**params)

                worker_stats = ray_xshards.zip_reduce_shards_with_actors(val_ray_xshards,
                                                                         self.remote_workers,
                                                                         zip_func)
        elif isinstance(data, Dataset):
            # todo: need to refactor to align with tf2 code
            params["wrap_dataloader"] = False
            shards = data.split(n=self.num_workers, locality_hints=self.remote_workers)

            def make_data_creator(shard, feature_cols, label_cols):
                def data_creator(config, batch_size):
                    torch_datashard = shard.to_torch(label_column=label_cols,
                                                     feature_columns=feature_cols,
                                                     batch_size=batch_size)
                    return torch_datashard
                return data_creator

            remote_worker_stats = []
            if validation_data is None:
                for shard, worker in zip(shards, self.remote_workers):
                    params["data_creator"] = make_data_creator(shard, feature_cols, label_cols)
                    stats = worker.train_epochs.remote(**params)
                    remote_worker_stats.append(stats)
            else:
                if self.backend == "horovod":
                    invalidInputError(False,
                                      "Currently, we don't support input validation_data for"
                                      " horovod backend")
                if not isinstance(validation_data, ray.data.Dataset):
                    invalidInputError(False,
                                      "Validation data type should be the same as train data,"
                                      " but got type: {}".format(type(validation_data)))

                val_shards = validation_data.split(n=self.num_workers,
                                                   locality_hints=self.remote_workers)
                for shard, val_shard, worker in zip(shards, val_shards, self.num_workers):
                    params["data_creator"] = make_data_creator(shard, feature_cols, label_cols)

                    params["validation_data_creator"] = make_data_creator(val_shard,
                                                                          feature_cols,
                                                                          label_cols)
                    stats = worker.train_epochs.remote(**params)
                    remote_worker_stats.append(stats)

            success = check_for_failure(remote_worker_stats)
            if success:
                worker_stats = ray.get(remote_worker_stats)
            else:
                worker_stats = None
        else:
            invalidInputError(isinstance(data, types.FunctionType),
                              "data should be either an instance of SparkXShards,"
                              " Ray Dataset or a callable function, but"
                              " got type: {}".format(type(data)))

            params["data_creator"] = data
            params["validation_data_creator"] = validation_data
            success, worker_stats = self._train_epochs(**params)

        epoch_stats = list(map(list, zip(*worker_stats)))
        if reduce_results:
            for i in range(len(epoch_stats)):
                epoch_stats[i] = self._process_stats(epoch_stats[i])
            return epoch_stats
        else:
            return epoch_stats

    def predict(self,
                data,
                batch_size=32,
                feature_cols=None,
                profile=False):
        """
        Using this PyTorch model to make predictions on the data.

        :param data: An instance of SparkXShards, a Ray Dataset or a Spark DataFrame
        :param batch_size: The number of samples per batch for each worker. Default is 32.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param feature_cols: feature column names if data is a Spark DataFrame or Ray Dataset.
        :return: A SparkXShards or a list that contains the predictions with key "prediction"
               in each shard
        """
        from bigdl.orca.data import SparkXShards
        param = dict(
            batch_size=batch_size,
            profile=profile
        )
        from pyspark.sql import DataFrame
        if isinstance(data, DataFrame):
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict")
            pred_shards = self._predict_spark_xshards(xshards, param)
            result = convert_predict_xshards_to_dataframe(data, pred_shards)
        elif isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols)
            pred_shards = self._predict_spark_xshards(data, param)
            result = update_predict_xshards(data, pred_shards)
        elif isinstance(data, ray.data.Dataset):
            shards = data.split(n=self.num_workers, locality_hints=self.remote_workers)

            def data_creator(config, batch_size):
                torch_datashard = shard.to_torch(feature_columns=feature_cols,
                                                 batch_size=batch_size)
                return torch_datashard

            remote_worker_stats = []
            for shard, worker in zip(shards, self.remote_workers):
                worker_stats = worker.predict.remote(data_creator, batch_size, profile)
                remote_worker_stats.append(worker_stats)
            result = ray.data.from_numpy(remote_worker_stats).map(
                lambda r: {"prediction_result": r["value"]})
        else:
            invalidInputError(False,
                              "Only xshards, Spark DataFrame or Ray Dataset"
                              " is supported for predict")

        return result

    def evaluate(self,
                 data,
                 batch_size=32,
                 num_steps=None,
                 profile=False,
                 info=None,
                 feature_cols=None,
                 label_cols=None):
        """
        Evaluates a PyTorch model given validation data.
        Note that only accuracy for classification with zero-based label is supported by
        default. You can override validate_batch in TrainingOperator for other metrics.
        Calls `TrainingOperator.validate()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Spark DataFrame, a Ray Dataset or a function
               that takes config and batch_size as argument and returns a PyTorch DataLoader for
               validation.
        :param batch_size: The number of samples per batch for each worker. Default is 32.
               The total batch size would be workers_per_node*num_nodes.
               If your validation data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader.
        :param num_steps: The number of batches to compute the validation results on. This
               corresponds to the number of times `TrainingOperator.validate_batch` is called.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param info: An optional dictionary that can be passed to the TrainingOperator
               for validate.
        :param feature_cols: feature column names if train data is Spark DataFrame or Ray Dataset.
        :param label_cols: label column names if train data is Spark DataFrame or Ray Dataset.

        :return: A dictionary of metrics for the given data, including validation accuracy and loss.
                You can also provide custom metrics by passing in a custom training_operator_cls
                when creating the Estimator.
        """
        from bigdl.orca.data import SparkXShards
        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers)
        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)
            from bigdl.orca.data.utils import process_spark_xshards
            ray_xshards = process_spark_xshards(data, self.num_workers)

            def transform_func(worker, partition_refs):
                data_creator = partition_refs_to_creator(partition_refs)
                # Should not wrap DistributedSampler on DataLoader for SparkXShards input.
                return worker.validate.remote(
                    data_creator, batch_size, num_steps, profile, info, False)

            worker_stats = ray_xshards.reduce_partitions_for_actors(self.remote_workers,
                                                                    transform_func)
        elif isinstance(data, ray.data.Dataset):
            shards = data.split(n=self.num_workers, locality_hints=self.remote_workers)

            def data_creator(config, batch_size):
                torch_datashard = shard.to_torch(label_column=label_cols,
                                                 feature_columns=feature_cols,
                                                 batch_size=batch_size)
                return torch_datashard

            remote_worker_stats = []
            for shard, worker in zip(shards, self.remote_workers):
                stats = worker.validate.remote(
                    data_creator, batch_size, num_steps, profile, info, False)
                remote_worker_stats.append(stats)
            worker_stats = ray.get(remote_worker_stats)
        else:
            invalidInputError(isinstance(data, types.FunctionType),
                              "data should be either an instance of SparkXShards or a callable"
                              " function, but got type: {}".format(type(data)))

            params = dict(data_creator=data, batch_size=batch_size, num_steps=num_steps,
                          profile=profile, info=info)

            worker_stats = ray.get([w.validate.remote(**params) for w in self.remote_workers])
        return self._process_stats(worker_stats)

    def get_model(self):
        """
        Returns the learned PyTorch model.

        :return: The learned PyTorch model.
        """
        state = self.get_state_dict()
        model = self.model_creator(self.config)
        model_state = state["models"][0]
        model.load_state_dict(model_state)
        return model.module if hasattr(model, "module") else model

    @enable_multi_fs_save
    def save(self, model_path):
        """
        Saves the Estimator state (including model and optimizer) to the provided model_path.

        :param model_path: (str) Path to save the model.
        :return:
        """
        state_dict = self.get_state_dict()
        torch.save(state_dict, model_path)
        return model_path

    @enable_multi_fs_load
    def load(self, model_path):
        """
        Loads the Estimator state (including model and optimizer) from the provided model_path.

        :param model_path: (str) Path to the existing model.
        """
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def save_checkpoint(self, model_path):
        """
        Manually saves the Estimator state (including model and optimizer) to the provided
        model_path.

        :param model_path: (str) Path to save the model. Both local and remote path are supported.
               e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"
        :return: None
        """
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(model_path):
            self.save(model_path)
        else:
            results = [
                worker.save_checkpoint.remote(model_path)
                for worker in self.remote_workers
            ]
            ray.get(results)

    def load_checkpoint(self, model_path):
        """
        Loads the Estimator state (including model and optimizer) from the provided model_path.

        :param model_path: (str) Path to the existing model. Both local and remote path are
               supported. e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"
        :return: None
        """
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(model_path):
            self.load(model_path)
        else:
            results = [
                worker.load_checkpoint.remote(model_path)
                for worker in self.remote_workers
            ]
            ray.get(results)

    def shutdown(self, force=False):
        """
        Shuts down workers and releases resources.

        :return:
        """
        if not force:
            cleanup = [
                worker.shutdown.remote() for worker in self.remote_workers
            ]
            try:
                ray.get(cleanup)
                [
                    worker.__ray_terminate__.remote()
                    for worker in self.remote_workers
                ]
            except RayActorError:
                logger.warning(
                    "Failed to shutdown gracefully, forcing a shutdown.")

                for worker in self.remote_workers:
                    logger.warning("Killing worker {}.".format(worker))
                    ray.kill(worker)
        else:
            for worker in self.remote_workers:
                logger.debug("Killing worker {}.".format(worker))
                ray.kill(worker)

        self.remote_workers = []

    def _process_stats(self, worker_stats):
        stats = {
            "num_samples": sum(
                stats.pop("num_samples", np.nan) for stats in worker_stats)
        }

        for stat_key in worker_stats[0]:
            if isinstance(worker_stats[0], numbers.Number):
                stats[stat_key] = np.nanmean(
                    [s.get(stat_key, np.nan) for s in worker_stats])
            else:
                stats[stat_key] = worker_stats[0][stat_key]
        return stats

    def _train_epochs(self, **params):
        remote_worker_stats = []
        for i, w in enumerate(self.remote_workers):
            stats = w.train_epochs.remote(**params)
            remote_worker_stats.append(stats)

        success = check_for_failure(remote_worker_stats)
        if success:
            return success, ray.get(remote_worker_stats)
        else:
            return success, None

    def _predict_spark_xshards(self, xshards, param):
        ray_xshards = RayXShards.from_spark_xshards(xshards)

        def transform_func(worker, shards_ref):
            data_creator = lambda config, batch_size: shards_ref
            return worker.predict.remote(
                data_creator, **param)

        pred_shards = ray_xshards.transform_shards_with_actors(self.remote_workers,
                                                               transform_func)
        spark_xshards = pred_shards.to_spark_xshards()
        return spark_xshards

    def get_state_dict(self):
        stream_ids = [
            worker.get_state_stream.remote()
            for worker in self.remote_workers
        ]
        # get the first task id that finished executing.
        [stream_id], stream_ids = ray.wait(stream_ids, num_returns=1, timeout=None)
        byte_obj = ray.get(stream_id)
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(
            _buffer,
            map_location="cpu")
        return state_dict

    def load_state_dict(self, state_dict, blocking=True):
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        state_stream = _buffer.getvalue()
        state_id = ray.put(state_stream)

        remote_calls = [
            worker.load_state_stream.remote(state_id)
            for worker in self.remote_workers
        ]
        if blocking:
            ray.get(remote_calls)

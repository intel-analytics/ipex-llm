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
from bigdl.orca.learn.pytorch.torch_pyspark_runner import TorchPysparkRunner
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, update_predict_xshards, \
    process_xshards_of_pandas_dataframe
from bigdl.orca import OrcaContext
from bigdl.orca.learn.base_estimator import BaseEstimator
from bigdl.dllib.utils.file_utils import enable_multi_fs_load, enable_multi_fs_save
from bigdl.dllib.utils.common import get_node_and_core_number


import ray
from ray.exceptions import RayActorError

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


class PyTorchPySparkEstimator(BaseEstimator):
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
            workers_per_node=1):
        if config is not None and "batch_size" in config:
            raise Exception("Please do not specify batch_size in config. Input batch_size in the"
                            " fit/evaluate/predict function of the estimator instead.")
        self.config = {} if config is None else config

        sc = OrcaContext.get_spark_context()
        if not (isinstance(model_creator, types.FunctionType) and
                isinstance(optimizer_creator, types.FunctionType)):  # Torch model is also callable.
            raise ValueError(
                "Must provide a function for both model_creator and optimizer_creator")

        if not training_operator_cls and not loss_creator:
            raise ValueError("If a loss_creator is not provided, you must "
                             "provide a custom training operator.")

        self.model_creator = model_creator
        self.initialization_hook = initialization_hook

        num_nodes, cores_per_node = get_node_and_core_number()
        self.num_workers = num_nodes * workers_per_node
        self.total_cores = num_nodes * cores_per_node
        self.cores_per_worker = cores_per_node // workers_per_node

        self.worker_init_params = dict(
            model_creator=self.model_creator,
            optimizer_creator=optimizer_creator,
            loss_creator=loss_creator,
            scheduler_creator=scheduler_creator,
            training_operator_cls=training_operator_cls,
            scheduler_step_freq=scheduler_step_freq,
            use_tqdm=use_tqdm,
            config=self.config.copy(),
            metrics=metrics,
            size=self.num_workers,
            cluster_info=self._get_cluster_info(sc)
        )

        # over partition to cover tasks all over the cluster
        self.workerRDD = sc.parallelize(list(range(self.total_cores * 4)),
                                        self.total_cores * 4).repartition(self.num_workers)

        self.model_weights = None

    def _get_cluster_info(self, sc):
        cluster_info = self.workerRDD.barrier().mapPartitions(find_ip_and_port).collect()
        return cluster_info

    def setup(self):
        RemoteRunner = ray.remote(num_cpus=self.cores_per_worker)(TorchPysparkRunner)
        self.remote_workers = [
            RemoteRunner.remote(**self.worker_init_params) for i in range(self.num_workers)
        ]
        ray.get([
            worker.setup.remote(self.cores_per_worker)
            for i, worker in enumerate(self.remote_workers)
        ])

        head_worker = self.remote_workers[0]
        address = ray.get(head_worker.setup_address.remote())

        logger.info(f"initializing pytorch process group on {address}")

        ray.get([
            worker.setup_torch_distribute.remote(address, i, self.num_workers)
            for i, worker in enumerate(self.remote_workers)
        ])

    def fit(self,
            data,
            epochs=1,
            batch_size=32,
            profile=False,
            reduce_results=True,
            info=None,
            feature_cols=None,
            label_cols=None):
        """
        Trains a PyTorch model given training data for several epochs.
        Calls `TrainingOperator.train_epoch()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Spark DataFrame or a function that
               takes config and batch_size as argument and returns a PyTorch DataLoader for
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
        :param feature_cols: feature column names if data is Spark DataFrame.
        :param label_cols: label column names if data is Spark DataFrame.

        :return: A list of dictionary of metrics for every training epoch. If reduce_results is
                False, this will return a nested list of metric dictionaries whose length will be
                equal to the total number of workers.
                You can also provide custom metrics by passing in a custom training_operator_cls
                when creating the Estimator.
        """
        from bigdl.orca.data import SparkXShards

        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="fit",
                                             num_workers=self.num_workers)

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)
            from bigdl.orca.data.utils import process_spark_xshards
            ray_xshards = process_spark_xshards(data, self.num_workers)

            def transform_func(worker, partition_refs):
                data_creator = partition_refs_to_creator(partition_refs)
                # Should not wrap DistributedSampler on DataLoader for SparkXShards input.
                return worker.train_epochs.remote(
                    data_creator, epochs, batch_size, profile, info, False)

            worker_stats = ray_xshards.reduce_partitions_for_actors(self.remote_workers,
                                                                    transform_func)
        else:
            assert isinstance(data, types.FunctionType), \
                "data should be either an instance of SparkXShards or a callable function, but " \
                "got type: {}".format(type(data))

            success, worker_stats = self._train_epochs(data,
                                                       epochs=epochs,
                                                       batch_size=batch_size,
                                                       profile=profile,
                                                       info=info)

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

        :param data: An instance of SparkXShards or a Spark DataFrame
        :param batch_size: The number of samples per batch for each worker. Default is 32.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param feature_cols: feature column names if data is a Spark DataFrame.
        :return: A SparkXShards that contains the predictions with key "prediction" in each shard
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
        else:
            raise ValueError("Only xshards or Spark DataFrame is supported for predict")

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

        :param data: An instance of SparkXShards, a Spark DataFrame or a function that
               takes config and batch_size as argument and returns a PyTorch DataLoader for
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
        :param feature_cols: feature column names if train data is Spark DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame.

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
        else:
            assert isinstance(data, types.FunctionType), \
                "data should be either an instance of SparkXShards or a callable function, but " \
                "got type: {}".format(type(data))

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

    def _train_epochs(self, data_creator, epochs=1, batch_size=32, profile=False, info=None):
        params = dict(data_creator=data_creator, epochs=epochs,
                      batch_size=batch_size, profile=profile, info=info)
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
            worker.state_stream.remote()
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

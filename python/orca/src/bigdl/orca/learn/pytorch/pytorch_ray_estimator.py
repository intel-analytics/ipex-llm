#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.orca.data.ray_xshards import RayXShards
from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.learn.pytorch.torch_runner import TorchRunner
from zoo.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, update_predict_xshards, \
    process_xshards_of_pandas_dataframe
from zoo.ray import RayContext

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
        from zoo.orca.data.utils import ray_partitions_get_data_label, index_data, get_size
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


class PyTorchRayEstimator:
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
            backend="torch_distributed",
            workers_per_node=1):

        # todo remove ray_ctx to run on workers
        ray_ctx = RayContext.get()
        if not (isinstance(model_creator, types.FunctionType) and
                isinstance(optimizer_creator, types.FunctionType)):  # Torch model is also callable.
            raise ValueError(
                "Must provide a function for both model_creator and optimizer_creator")

        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.scheduler_creator = scheduler_creator
        self.training_operator_cls = training_operator_cls
        self.scheduler_step_freq = scheduler_step_freq
        self.use_tqdm = use_tqdm

        if not training_operator_cls and not loss_creator:
            raise ValueError("If a loss_creator is not provided, you must "
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
            metrics=metrics
        )

        if backend == "torch_distributed":
            cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            num_nodes = ray_ctx.num_ray_nodes * workers_per_node
            RemoteRunner = ray.remote(num_cpus=cores_per_node)(TorchRunner)
            self.remote_workers = [
                RemoteRunner.remote(**params) for i in range(num_nodes)
            ]
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            head_worker = self.remote_workers[0]
            address = ray.get(head_worker.setup_address.remote())

            logger.info(f"initializing pytorch process group on {address}")

            ray.get([
                worker.setup_torch_distribute.remote(address, i, num_nodes)
                for i, worker in enumerate(self.remote_workers)
            ])

        elif backend == "horovod":
            from zoo.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
            self.horovod_runner = HorovodRayRunner(ray_ctx,
                                                   worker_cls=TorchRunner,
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
            raise Exception("Only \"torch_distributed\" and \"horovod\" are supported "
                            "values of backend, but got {}".format(backend))
        self.num_workers = len(self.remote_workers)

    def train(self,
              data,
              epochs=1,
              batch_size=32,
              profile=False,
              reduce_results=True,
              info=None,
              feature_cols=None,
              label_cols=None):
        """
        See the documentation in
        'zoo.orca.learn.pytorch.estimator.PyTorchRayEstimatorWrapper.fit'.
        """
        from zoo.orca.data import SparkXShards

        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="fit",
                                             num_workers=self.num_workers)

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)
            from zoo.orca.data.utils import process_spark_xshards
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

    def validate(self,
                 data,
                 batch_size=32,
                 num_steps=None,
                 profile=False,
                 info=None,
                 feature_cols=None,
                 label_cols=None):
        """
        See the documentation in
        'zoo.orca.learn.pytorch.estimator.PyTorchRayEstimatorWrapper.evaluate'.
        """
        from zoo.orca.data import SparkXShards
        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers)
        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)
            from zoo.orca.data.utils import process_spark_xshards
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

    def predict(self,
                data,
                batch_size=32,
                feature_cols=None,
                profile=False):
        from zoo.orca.data import SparkXShards
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

    def get_model(self):
        """Returns the learned model(s)."""
        state = self.get_state_dict()
        model = self.model_creator(self.config)
        model_state = state["models"][0]
        model.load_state_dict(model_state)
        return model.module if hasattr(model, "module") else model

    def save(self, model_path):
        """Saves the Estimator state to the provided model_path.

        :param model_path: (str) Path to save the model.
        """
        state_dict = self.get_state_dict()
        torch.save(state_dict, model_path)
        return model_path

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

    def load(self, model_path):
        """Loads the Estimator and all workers from the provided model_path.

        :param model_path: (str) Path to the existing model.
        """
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

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

    def shutdown(self, force=False):
        """Shuts down workers and releases resources."""
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

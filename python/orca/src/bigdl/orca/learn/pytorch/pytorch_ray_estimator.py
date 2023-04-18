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

import types
import copy
import logging
import math

from bigdl.orca.data.ray_xshards import RayXShards
from bigdl.orca.learn.pytorch.pytorch_ray_worker import PytorchRayWorker
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, update_predict_xshards, \
    process_xshards_of_pandas_dataframe, add_predict_to_pd_xshards
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca.learn.pytorch.core.base_ray_estimator import BaseRayEstimator
from bigdl.orca.learn.pytorch.utils import process_stats, check_for_failure
from bigdl.orca.learn.pytorch.callbacks.maincallback import make_only_mainCallback
from bigdl.orca.learn.pytorch.callbacks.tqdm import TqdmCallback, is_tqdm_exists
from bigdl.orca.learn.pytorch.callbacks.maxsteps import MaxstepsCallback

import ray
from bigdl.dllib.utils.log4Error import invalidInputError

from typing import TYPE_CHECKING, Union, Optional, Callable, Dict, List
if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer
    from bigdl.orca.learn.metrics import Metric
    from torch.nn.modules.loss import _Loss as Loss
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
    from torch.distributed import TCPStore
    from torch.utils.data import DataLoader
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame
    from ray.data import Dataset as RayDataset
    from bigdl.orca.learn.pytorch.callbacks import Callback
    from bigdl.orca.data import SparkXShards


def partition_refs_to_creator(partition_refs):
    def data_creator(config, batch_size):
        from bigdl.orca.data.utils import partitions_get_data_label, index_data, get_size
        from torch.utils.data import Dataset, DataLoader

        class NDArrayDataset(Dataset):
            def __init__(self, x, y):
                self.x = x  # features
                self.y = y  # labels

            def __len__(self):
                return get_size(self.y)

            def __getitem__(self, i):
                index_data_x = index_data(self.x, i)
                if isinstance(index_data_x, (list, tuple)):
                    return (*index_data_x, index_data(self.y, i))
                else:
                    return (index_data_x, index_data(self.y, i))

        params = {"batch_size": batch_size, "shuffle": True}
        for arg in ["shuffle", "sampler", "batch_sampler", "num_workers", "collate_fn",
                    "pin_memory", "drop_last", "timeout", "worker_init_fn",
                    "multiprocessing_context"]:
            if arg in config:
                params[arg] = config[arg]
        data, label = partitions_get_data_label(ray.get(partition_refs),
                                                allow_tuple=False,
                                                allow_list=False)
        print("Data size on worker: ", len(label))
        dataset = NDArrayDataset(data, label)
        data_loader = DataLoader(dataset, **params)
        return data_loader

    return data_creator


class PyTorchRayEstimator(BaseRayEstimator):
    def __init__(
            self,
            *,
            model_creator: Union[Callable[[Dict], 'Module'], None],
            optimizer_creator: Union[Callable[['Module', Dict], 'Optimizer'],
                                     None]=None,
            loss_creator: Union['Loss', Callable[[Dict], 'Loss'], None]=None,
            metrics: Union['Metric', List['Metric'], None]=None,
            scheduler_creator: Optional[Callable[[Dict], 'LRScheduler']]=None,
            config: Dict=None,
            use_tqdm: bool=False,
            backend: str="ray",
            workers_per_node: int=1,
            sync_stats: bool=True,
            log_level: int=logging.INFO):
        if config is not None and "batch_size" in config:
            invalidInputError(False,
                              "Please do not specify batch_size in config. Input batch_size in the"
                              " fit/evaluate/predict function of the estimator instead.")

        # todo remove ray_ctx to run on workers
        ray_ctx = OrcaRayContext.get()
        if not isinstance(model_creator, types.FunctionType):  # Torch model is also callable.
            invalidInputError(False,
                              "Must provide a function for model_creator.")

        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.scheduler_creator = scheduler_creator
        self.sync_stats = sync_stats
        self.use_tqdm = use_tqdm
        self.backend = backend
        self.workers_per_node = workers_per_node

        self.config = {} if config is None else config
        worker_config = copy.copy(self.config)
        self.setup_params = dict(
            model_creator=self.model_creator,
            optimizer_creator=self.optimizer_creator,
            loss_creator=self.loss_creator,
            scheduler_creator=self.scheduler_creator,
            config=worker_config,
            metrics=metrics,
            sync_stats=sync_stats,
            log_level=log_level
        )
        self.setup(params=self.setup_params,
                   backend=self.backend,
                   runner_cls=PytorchRayWorker,
                   workers_per_node=self.workers_per_node)

    def fit(self,
            data: Union['SparkXShards',
                        'SparkDataFrame',
                        'RayDataset',
                        Callable[[Dict, int], 'DataLoader']],
            epochs: int=1,
            max_steps: Optional[int] = None,
            batch_size: int=32,
            profile: bool=False,
            reduce_results: bool=True,
            feature_cols: Optional[List[str]]=None,
            label_cols: Optional[List[str]]=None,
            validation_data: Union['SparkXShards',
                                   'SparkDataFrame',
                                   Callable[[Dict, int], 'DataLoader'],
                                   None]=None,
            callbacks: Optional[List['Callback']]=None) -> List:
        """
        Trains a PyTorch model given training data for several epochs.
        Calls `TorchRunner.train_epoch()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Ray Dataset, a Spark DataFrame or a function
               that takes config and batch_size as argument and returns a PyTorch DataLoader for
               training.
        :param epochs: The number of epochs to train the model. Default is 1.
        :param max_steps: The max steps to train the model. Default is None.
         If max_steps > 0, `epochs` would be ignored.
        :param batch_size: Total batch size for all workers used for training. Each worker's batch
               size would be this value divide the total number of workers. Default is 32.
               If your training data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param reduce_results: Boolean. Whether to average all metrics across all workers into
               one dict. If a metric is a non-numerical value, the one value will be randomly
               selected among the workers. If False, returns a list of dicts for
               all workers. Default is True.
        :param feature_cols: feature column names if data is Spark DataFrame or Ray Dataset.
        :param label_cols: label column names if data is Spark DataFrame or Ray Dataset.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param callbacks: A list for all callbacks. Note that only one MainCallback
               is allowed among all callbacks.

        :return: A list of dictionary of metrics for every training epoch. If reduce_results is
                False, this will return a nested list of metric dictionaries whose length will be
                equal to the total number of workers.
                You can also provide custom metrics by passing in a custom HookClass(after 2.2.0)
                when creating the Estimator.
        """
        invalidInputError(isinstance(batch_size, int) and batch_size > 0,
                          "batch_size should be a positive integer")
        batch_size = batch_size // self.num_workers  # Local batch size for each worker
        if batch_size <= 0:
            batch_size = 1

        # Check uniqueness of the MainCallback
        callbacks = callbacks or []
        make_only_mainCallback(callbacks)
        if self.use_tqdm and not is_tqdm_exists(callbacks):
            callbacks.append(TqdmCallback())
        if max_steps is not None:
            callbacks.append(MaxstepsCallback(max_step=max_steps))

        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            profile=profile,
            callbacks=callbacks,
        )

        if self.backend == "ray" and not self.init_ddp_process:
            self.setup_torch_ddp()

        from bigdl.orca.data import SparkXShards
        from ray.data import Dataset

        data, validation_data = maybe_dataframe_to_xshards(data,
                                                           validation_data=validation_data,
                                                           feature_cols=feature_cols,
                                                           label_cols=label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers,
                                                           shard_size=batch_size)

        if isinstance(data, SparkXShards):
            # Should not wrap DistributedSampler on DataLoader for SparkXShards input.
            params["wrap_dataloader"] = False
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data, "fit")
            from bigdl.orca.data.utils import process_spark_xshards
            ray_xshards = process_spark_xshards(data, self.num_workers)  # type:ignore

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
                val_ray_xshards = process_spark_xshards(validation_data,  # type:ignore
                                                        self.num_workers)

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

                val_shards = validation_data.split(n=self.num_workers,  # type:ignore
                                                   locality_hints=self.remote_workers)
                for shard, val_shard, worker in zip(shards,
                                                    val_shards,  # type:ignore
                                                    self.num_workers):
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

        epoch_stats = list(map(list, zip(*worker_stats)))  # type:ignore
        if reduce_results:
            for i in range(len(epoch_stats)):
                epoch_stats[i] = process_stats(epoch_stats[i])
            return epoch_stats
        else:
            return epoch_stats

    def predict(self,
                data: Union['SparkXShards', 'SparkDataFrame'],
                batch_size: int=32,
                feature_cols: Optional[List[str]]=None,
                profile: bool=False,
                callbacks: Optional[List['Callback']]=None) -> Union['SparkXShards',
                                                                     'SparkDataFrame']:
        """
        Using this PyTorch model to make predictions on the data.

        :param data: An instance of SparkXShards, a Ray Dataset or a Spark DataFrame
        :param batch_size: Total batch size for all workers used for inference. Each worker's batch
               size would be this value divide the total number of workers. Default is 32.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param feature_cols: feature column names if data is a Spark DataFrame or Ray Dataset.
        :return: A SparkXShards or a list that contains the predictions with key "prediction"
               in each shard
        """
        invalidInputError(isinstance(batch_size, int) and batch_size > 0,
                          "batch_size should be a positive integer")
        batch_size = batch_size // self.num_workers  # Local batch size for each worker
        if batch_size <= 0:
            batch_size = 1
        from bigdl.orca.data import SparkXShards

        callbacks = callbacks or []
        make_only_mainCallback(callbacks)
        if self.use_tqdm:
            callbacks.append(TqdmCallback())

        params = dict(
            batch_size=batch_size,
            profile=profile,
            callbacks=callbacks
        )

        from pyspark.sql import DataFrame
        if isinstance(data, DataFrame):
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict",
                                              shard_size=batch_size)
            pred_shards = self._predict_spark_xshards(xshards, params)
            result = convert_predict_xshards_to_dataframe(data, pred_shards)
        elif isinstance(data, SparkXShards):
            xshards = data.to_lazy()
            if xshards._get_class_name() == 'pandas.core.frame.DataFrame':
                xshards = process_xshards_of_pandas_dataframe(xshards, feature_cols)
                pred_shards = self._predict_spark_xshards(xshards, params)
                result = add_predict_to_pd_xshards(data, pred_shards)
            else:
                pred_shards = self._predict_spark_xshards(xshards, params)
                result = update_predict_xshards(data, pred_shards)
        elif isinstance(data, ray.data.Dataset):
            shards = data.split(n=self.num_workers, locality_hints=self.remote_workers)

            def data_creator(config, batch_size):
                torch_datashard = shard.to_torch(feature_columns=feature_cols,
                                                 batch_size=batch_size)
                return torch_datashard

            remote_worker_stats = []
            for shard, worker in zip(shards, self.remote_workers):
                worker_stats = worker.predict.remote(data_creator, batch_size, profile, callbacks)
                remote_worker_stats.append(worker_stats)
            result = ray.data.from_numpy(remote_worker_stats).map(
                lambda r: {"prediction_result": r["value"]})
        else:
            invalidInputError(False,
                              "Only xshards, Spark DataFrame or Ray Dataset"
                              " is supported for predict")

        return result

    def evaluate(self,
                 data: Union['SparkXShards',
                             'SparkDataFrame',
                             'RayDataset',
                             Callable[[Dict, int], 'DataLoader']],
                 batch_size: int=32,
                 num_steps: int=None,
                 profile: bool=False,
                 reduce_results: bool=True,
                 feature_cols: Optional[List[str]]=None,
                 label_cols:  Optional[List[str]]=None,
                 callbacks: Optional[List['Callback']]=None) -> Union[List[Dict], Dict]:
        """
        Evaluates a PyTorch model given validation data.
        Note that only accuracy for classification with zero-based label is supported by
        default. You can override validate_batch in TorchRunner for other metrics.
        Calls `TorchRunner.validate()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Spark DataFrame, a Ray Dataset or a function
               that takes config and batch_size as argument and returns a PyTorch DataLoader for
               validation.
        :param batch_size: Total batch size for all workers used for evaluation. Each worker's batch
               size would be this value divide the total number of workers. Default: 32.
               If your validation data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader.
        :param num_steps: The number of batches to compute the validation results on. This
               corresponds to the number of times `TorchRunner.validate_batch` is called.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param reduce_results: Boolean. Whether to average all metrics across all workers into
               one dict. If a metric is a non-numerical value, the one value will be randomly
               selected among the workers. If False, returns a list of dicts for
               all workers. Default is True.
        :param feature_cols: feature column names if train data is Spark DataFrame or Ray Dataset.
        :param label_cols: label column names if train data is Spark DataFrame or Ray Dataset.
        :param callbacks: A list for all callbacks. Note that only one MainCallback
               is allowed among all callbacks.

        :return: A dictionary of metrics for the given data, including validation accuracy and loss.
                You can also provide custom metrics by passing in a custom HookClass(after 2.2.0)
                when creating the Estimator.
        """
        invalidInputError(isinstance(batch_size, int) and batch_size > 0,
                          "batch_size should be a positive integer")
        batch_size = batch_size // self.num_workers  # Local batch size for each worker
        if batch_size <= 0:
            batch_size = 1
        from bigdl.orca.data import SparkXShards
        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers,
                                             shard_size=batch_size)

        # Check uniqueness of the MainCallback
        callbacks = callbacks or []
        make_only_mainCallback(callbacks)
        if self.use_tqdm and not is_tqdm_exists(callbacks):
            callbacks.append(TqdmCallback())

        params = dict(batch_size=batch_size,
                      num_steps=num_steps,
                      profile=profile,
                      wrap_dataloader=False,
                      callbacks=callbacks)

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)
            from bigdl.orca.data.utils import process_spark_xshards
            ray_xshards = process_spark_xshards(data, self.num_workers)  # type:ignore

            def transform_func(worker, partition_refs):
                data_creator = partition_refs_to_creator(partition_refs)
                # Should not wrap DistributedSampler on DataLoader for SparkXShards input.
                params["data_creator"] = data_creator
                return worker.validate.remote(**params)

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
            params["data_creator"] = data_creator  # type:ignore
            for shard, worker in zip(shards, self.remote_workers):
                stats = worker.validate.remote(**params)
                remote_worker_stats.append(stats)
            worker_stats = ray.get(remote_worker_stats)
        else:
            invalidInputError(isinstance(data, types.FunctionType),
                              "data should be either an instance of SparkXShards or a callable"
                              " function, but got type: {}".format(type(data)))

            params["data_creator"] = data  # type:ignore
            worker_stats = ray.get([w.validate.remote(**params) for w in self.remote_workers])

        if reduce_results:
            return process_stats(worker_stats)
        else:
            return worker_stats

    def get_model(self) -> 'Module':
        """
        Returns the learned PyTorch model.

        :return: The learned PyTorch model.
        """
        state = self.get_state_dict()
        model = self.model_creator(self.config)  # type:ignore
        model_state = state["models"][0]
        model.load_state_dict(model_state)
        return model.module if hasattr(model, "module") else model  # type:ignore

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

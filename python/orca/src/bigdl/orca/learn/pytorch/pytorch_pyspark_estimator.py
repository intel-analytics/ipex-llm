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
import torch
import copy
import os
import shutil
import tempfile
import logging

from pyspark.sql.dataframe import DataFrame

from bigdl.dllib.utils.file_utils import is_local_path
from bigdl.orca.learn.pytorch.pytorch_pyspark_worker import PytorchPysparkWorker
from bigdl.orca.learn.pytorch.utils import process_stats
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, make_data_creator, update_predict_xshards, \
    reload_dataloader_creator, process_xshards_of_pandas_dataframe, add_predict_to_pd_xshards
from bigdl.orca.data import SparkXShards
from bigdl.orca import OrcaContext
from bigdl.orca.learn.base_estimator import BaseEstimator
from bigdl.orca.data.file import get_remote_file_to_local, put_local_file_to_remote
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.orca.learn.log_monitor import start_log_server, stop_log_server
from bigdl.orca.learn.pytorch.callbacks.maincallback import make_only_mainCallback
from bigdl.orca.learn.pytorch.callbacks.tqdm import TqdmCallback, is_tqdm_exists
from bigdl.orca.learn.utils import find_free_port, find_ip_and_free_port
from bigdl.dllib.utils.utils import get_node_ip
from bigdl.dllib.utils.log4Error import invalidInputError

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

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
    from pyspark.rdd import RDD

from bigdl.orca.learn.pytorch.callbacks import Callback


def partition_to_creator(partition):
    def data_creator(config, batch_size):
        from bigdl.orca.data.utils import partition_get_data_label, index_data, get_size
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
        data, label = partition_get_data_label(partition,
                                               allow_tuple=False,
                                               allow_list=False)
        print("Data size on worker: ", len(label))
        dataset = NDArrayDataset(data, label)
        data_loader = DataLoader(dataset, **params)
        return data_loader

    return data_creator


def parse_model_dir(model_dir):
    if model_dir and model_dir.startswith("dbfs:/"):
        model_dir = "/dbfs/" + model_dir[len("dbfs:/"):]
    return model_dir


class PyTorchPySparkEstimator(BaseEstimator):
    def __init__(
            self,
            *,
            model_creator: Union[Callable[[Dict], 'Module'], None]=None,
            optimizer_creator: Union[Callable[['Module', Dict], 'Optimizer'],
                                     None]=None,
            loss_creator: Union['Loss', Callable[[Dict], 'Loss'], None]=None,
            metrics: Union['Metric', List['Metric'], None]=None,
            scheduler_creator: Optional[Callable[[Dict], 'LRScheduler']]=None,
            config: Optional[Dict]=None,
            use_tqdm: bool=False,
            workers_per_node: int=1,
            sync_stats: bool=True,
            log_level: int=logging.INFO,
            model_dir: Optional[str]=None,
            log_to_driver: bool=True):
        logging.basicConfig(level=log_level,
                            format='[%(asctime)s] %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                            )
        self.logger = logging.getLogger(__name__)
        if config is not None and "batch_size" in config:
            invalidInputError(False,
                              "Please do not specify batch_size in config. Input batch_size in the"
                              " fit/evaluate/predict function of the estimator instead.")
        self.config = {} if config is None else config

        sc = OrcaContext.get_spark_context()

        if model_creator and not isinstance(model_creator, types.FunctionType):
            # Torch model is also callable.
            invalidInputError(False,
                              "Must provide a function for model_creator")

        self.model_dir = parse_model_dir(model_dir)
        self.use_tqdm = use_tqdm

        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator

        num_nodes, cores_per_node = get_node_and_core_number()
        self.num_workers = num_nodes * workers_per_node
        self.total_cores = num_nodes * cores_per_node
        self.cores_per_worker = cores_per_node // workers_per_node

        # over partition to cover tasks all over the cluster
        self.workerRDD = sc.parallelize(list(range(self.total_cores * 4)),
                                        self.total_cores * 4).repartition(self.num_workers)

        self.ip = get_node_ip()
        self.log_port = find_free_port()
        is_local = sc.master.startswith("local")
        self.need_to_log_to_driver = (not is_local) and log_to_driver
        if self.need_to_log_to_driver:
            self.log_server_thread = start_log_server(self.ip, self.log_port)
        self.tcp_store_port = find_free_port()

        self.worker_init_params = dict(
            model_creator=self.model_creator,
            optimizer_creator=optimizer_creator,
            loss_creator=loss_creator,
            scheduler_creator=scheduler_creator,
            config=copy.copy(self.config),
            metrics=metrics,
            size=self.num_workers,
            cores_per_worker=self.cores_per_worker,
            sync_stats=sync_stats,
            log_level=log_level,
            model_dir=self.model_dir,
            log_to_driver=self.need_to_log_to_driver,
            driver_ip=self.ip,
            driver_log_port=self.log_port,
            driver_tcp_store_port=self.tcp_store_port)

        local_init_params = self.worker_init_params.copy()
        local_init_params["log_to_driver"] = False
        self.driver_runner = PytorchPysparkWorker(
            mode='predict',
            cluster_info=self._get_cluster_info(sc),
            **local_init_params)

        if self.model_creator:
            self.state_dict = self.driver_runner.get_state_dict()

    def create_tcpstore_server(self) -> 'TCPStore':
        import torch.distributed as dist
        server_store = dist.TCPStore(self.ip, self.tcp_store_port, -1, True,
                                     dist.constants.default_pg_timeout)
        return server_store

    def _get_cluster_info(self, sc):
        cluster_info = self.workerRDD.barrier().mapPartitions(find_ip_and_free_port).collect()
        return cluster_info

    def fit(self,
            data: Union['SparkXShards', 'SparkDataFrame', Callable[[Dict, int], 'DataLoader']],
            epochs: int=1,
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
        Calls `TorchRunner.train_epochs()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Spark DataFrame or a function that
               takes config and batch_size as argument and returns a PyTorch DataLoader for
               training.
        :param epochs: The number of epochs to train the model. Default is 1.
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
        :param feature_cols: feature column names if data is Spark DataFrame.
        :param label_cols: label column names if data is Spark DataFrame.
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
        if isinstance(data, SparkXShards):
            data = data.to_lazy()
            if validation_data is not None and isinstance(validation_data, SparkXShards):
                validation_data = validation_data.to_lazy()
        # Data partition should be equal to num workers.
        # Repartition Spark DataFrame before converting to SparkXShards.
        # Repartition on SparkXShards will result in empty partitions.
        if isinstance(data, DataFrame) or isinstance(data, SparkXShards):
            if data.rdd.getNumPartitions() != self.num_workers:
                data = data.repartition(self.num_workers)
            if validation_data is not None:
                invalidInputError(
                    isinstance(validation_data, DataFrame) or
                    isinstance(validation_data, SparkXShards),
                    "validation_data should have the same type with train data")
                if validation_data.rdd.getNumPartitions() != self.num_workers:  # type:ignore
                    validation_data = validation_data.repartition(self.num_workers)  # type:ignore
        data, validation_data = maybe_dataframe_to_xshards(data,
                                                           validation_data=validation_data,
                                                           feature_cols=feature_cols,
                                                           label_cols=label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers,
                                                           shard_size=batch_size)

        if self.model_creator is None:
            invalidInputError(False,
                              "Must provide callable function for model_creator "
                              "or load a saved model.")

        sc = OrcaContext.get_spark_context()
        _ = self.create_tcpstore_server()
        cluster_info = self._get_cluster_info(sc)
        state_dict = self._get_broadcasted_state_dict(sc)
        init_params = dict(
            mode="fit",
            state_dict=state_dict,
            cluster_info=cluster_info)
        init_params.update(self.worker_init_params)

        # Check uniqueness of the MainCallback
        callbacks = callbacks or []
        make_only_mainCallback(callbacks)
        if self.use_tqdm and not is_tqdm_exists(callbacks):
            callbacks.append(TqdmCallback())

        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            profile=profile,
            callbacks=callbacks
        )

        if not isinstance(self.optimizer_creator, types.FunctionType):
            invalidInputError(False,
                              "Must provide a function for optimizer_creator")

        if isinstance(data, SparkXShards):  # Computation triggered when collect
            params["wrap_dataloader"] = False
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data, "fit")

            if validation_data is None:
                def transform_func(iter, init_params, param):
                    partition_data = list(iter)
                    param["data_creator"] = partition_to_creator(partition_data)
                    runner = PytorchPysparkWorker(**init_params)
                    result = runner.train_epochs(**param)
                    runner.shutdown()
                    return result

                data_rdd = data.rdd  # type:ignore
                res = data_rdd.barrier().mapPartitions(
                    lambda iter: transform_func(iter, init_params, params)).collect()

            else:
                def transform_func(iter, init_params, param):
                    data_tuple_list = list(iter)
                    data_list = [x for data_tuple in data_tuple_list for x in data_tuple[0]]
                    valid_list = [x for data_tuple in data_tuple_list for x in data_tuple[1]]
                    param["data_creator"] = partition_to_creator(data_list)
                    param["validation_data_creator"] = partition_to_creator(valid_list)
                    runner = PytorchPysparkWorker(**init_params)
                    result = runner.train_epochs(**param)
                    runner.shutdown()
                    return result

                train_rdd = data.rdd.mapPartitions(lambda iter: [list(iter)])  # type:ignore
                val_rdd = validation_data.rdd  # type:ignore
                val_rdd = val_rdd.mapPartitions(lambda iter: [list(iter)])
                res = train_rdd.zip(val_rdd).barrier().mapPartitions(
                    lambda iter: transform_func(iter, init_params, params)).collect()

        else:
            if not isinstance(data, types.FunctionType):
                invalidInputError(False,
                                  "data should be either an instance of SparkXShards or a "
                                  "callable  function, but got type: {}".format(type(data)))

            params["data_creator"] = reload_dataloader_creator(data)
            params["validation_data_creator"] = reload_dataloader_creator(validation_data)

            def transform_func(iter, init_param, param):  # type:ignore
                return PytorchPysparkWorker(**init_param).train_epochs(**param)

            res = self.workerRDD.barrier().mapPartitions(
                lambda iter: transform_func(iter, init_params, params)).collect()

        if self.model_dir is not None:
            self.state_dict = PyTorchPySparkEstimator._get_state_dict_from_remote(self.model_dir)
            worker_stats = res
        else:
            self.state_dict = res[0]  # state dicts of all runners would be the same
            # Each runner would return a list of worker stats for different epochs
            worker_stats = [item for item in res if isinstance(item, list)]

        epoch_stats = list(map(list, zip(*worker_stats)))
        if reduce_results:
            for i in range(len(epoch_stats)):
                epoch_stats[i] = process_stats(epoch_stats[i])
            return epoch_stats
        else:
            return epoch_stats

    @staticmethod
    def _get_state_dict_from_remote(remote_dir):
        import tempfile
        import shutil
        import os
        try:
            temp_dir = tempfile.mkdtemp()
            get_remote_file_to_local(os.path.join(remote_dir, "state.pkl"),
                                     os.path.join(temp_dir, "state.pkl"))
            import pickle
            with open(os.path.join(temp_dir, "state.pkl"), 'rb') as f:
                state_dicts = pickle.load(f)
        finally:
            shutil.rmtree(temp_dir)
        return state_dicts

    def _get_broadcasted_state_dict(self, sc):
        if self.state_dict:
            state_dict_b = sc.broadcast(self.state_dict)
        else:
            state_dict_b = None
        return state_dict_b

    def _predict_spark_xshards(self, xshards, init_params, params):
        def transform_func(iter, init_param, param):
            partition_data = list(iter)
            # res = combine_in_partition(partition_data)
            param["data_creator"] = make_data_creator(partition_data)
            return PytorchPysparkWorker(**init_param).predict(**params)

        pred_shards = SparkXShards.lazy(xshards.rdd.mapPartitions(
            lambda iter: transform_func(iter, init_params, params)))
        return pred_shards

    def predict(self,
                data: Union['SparkXShards', 'SparkDataFrame'],
                batch_size: int=32,
                feature_cols: Optional[List[str]]=None,
                profile: bool=False,
                callbacks: Optional[List['Callback']]=None) -> Union['SparkXShards',
                                                                     'SparkDataFrame']:
        """
        Using this PyTorch model to make predictions on the data.

        :param data: An instance of SparkXShards or a Spark DataFrame
        :param batch_size: Total batch size for all workers used for inference. Each worker's batch
               size would be this value divide the total number of workers. Default is 32.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param feature_cols: feature column names if data is a Spark DataFrame.
        :return: A SparkXShards that contains the predictions with key "prediction" in each shard
        """
        invalidInputError(isinstance(batch_size, int) and batch_size > 0,
                          "batch_size should be a positive integer")
        batch_size = batch_size // self.num_workers  # Local batch size for each worker
        if batch_size <= 0:
            batch_size = 1

        if self.model_creator is None:
            invalidInputError(False,
                              "Must provide callable function for model_creator "
                              "or load a saved model.")

        sc = OrcaContext.get_spark_context()
        cluster_info = self._get_cluster_info(sc)
        state_dict = self._get_broadcasted_state_dict(sc)

        init_params = dict(
            mode="predict",
            state_dict=state_dict,
            cluster_info=cluster_info)
        init_params.update(self.worker_init_params)

        callbacks = callbacks or []
        make_only_mainCallback(callbacks)
        if self.use_tqdm:
            callbacks.append(TqdmCallback())

        params = dict(
            batch_size=batch_size,
            profile=profile,
            callbacks=callbacks
        )

        if isinstance(data, DataFrame):  # Computation would be triggered by the user
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict",
                                              shard_size=batch_size)

            pred_shards = self._predict_spark_xshards(xshards, init_params, params)
            result = convert_predict_xshards_to_dataframe(data, pred_shards)
        elif isinstance(data, SparkXShards):  # Computation triggered when updating XShards
            xshards = data.to_lazy()
            if xshards._get_class_name() == 'pandas.core.frame.DataFrame':
                xshards = process_xshards_of_pandas_dataframe(xshards, feature_cols)
                pred_shards = self._predict_spark_xshards(xshards, init_params, params)
                # Should add to the original SparkXShards of Pandas DataFrames
                result = add_predict_to_pd_xshards(data, pred_shards)
            else:
                pred_shards = self._predict_spark_xshards(xshards, init_params, params)
                result = update_predict_xshards(data, pred_shards)
            # Uncache the original data since it is already included in the result
            data.uncache()
        else:
            invalidInputError(False,
                              "Only XShards or Spark DataFrame are supported for predict")

        return result

    def evaluate(self,
                 data: Union['SparkXShards',
                             'SparkDataFrame',
                             Callable[[Dict, int], 'DataLoader']],
                 batch_size: int=32,
                 num_steps: Optional[int]=None,
                 profile: bool=False,
                 reduce_results: bool=True,
                 feature_cols: Optional[List[str]]=None,
                 label_cols: Optional[List[str]]=None,
                 callbacks: Optional[List['Callback']]=None) -> Union[List[Dict], Dict]:
        """
        Evaluates a PyTorch model given validation data.
        Note that only accuracy for classification with zero-based label is supported by
        default. You can override validate_batch in TorchRunner for other metrics.
        Calls `TorchRunner.validate()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Spark DataFrame or a function that
               takes config and batch_size as argument and returns a PyTorch DataLoader for
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
        :param feature_cols: feature column names if train data is Spark DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame.
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

        if self.model_creator is None:
            invalidInputError(False,
                              "Must provide callable function for model_creator "
                              "or load a saved model.")

        sc = OrcaContext.get_spark_context()
        cluster_info = self._get_cluster_info(sc)
        state_dict = self._get_broadcasted_state_dict(sc)
        init_params = dict(
            mode="evaluate",
            state_dict=state_dict,
            cluster_info=cluster_info)
        init_params.update(self.worker_init_params)

        # Check uniqueness of the MainCallback
        callbacks = callbacks or []
        make_only_mainCallback(callbacks)
        if self.use_tqdm and not is_tqdm_exists(callbacks):
            callbacks.append(TqdmCallback())

        params = dict(
            batch_size=batch_size,
            num_steps=num_steps,
            profile=profile,
            callbacks=callbacks
        )

        if isinstance(data, SparkXShards):
            data = data.to_lazy()
        if isinstance(data, DataFrame) or isinstance(data, SparkXShards):
            if data.rdd.getNumPartitions() != self.num_workers:
                data = data.repartition(self.num_workers)
        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers,
                                             shard_size=batch_size)
        if isinstance(data, SparkXShards):  # Computation triggered when collect
            params["wrap_dataloader"] = False
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)

            def transform_func(iter, init_param, param):
                partition_data = list(iter)
                param["data_creator"] = partition_to_creator(partition_data)
                return PytorchPysparkWorker(**init_param).validate(**param)

            data_rdd = data.rdd  # type:ignore
            res = data_rdd.barrier().mapPartitions(
                lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params["data_creator"] = reload_dataloader_creator(data)

            def transform_func(iter, init_param, param):
                return PytorchPysparkWorker(**init_param).validate(**param)

            res = self.workerRDD.barrier().mapPartitions(
                lambda iter: transform_func(iter, init_params, params)).collect()

        if reduce_results:
            return process_stats(res)
        else:
            return res

    def get_model(self) -> 'Module':
        """
        Returns the learned PyTorch model.

        :return: The learned PyTorch model.
        """
        if self.model_creator:
            state = self.state_dict
            model = self.model_creator(self.config)
            model_state = state["models"][0]
            model.load_state_dict(model_state)
        else:
            invalidInputError(False,
                              "Must provide callable function for model_creator "
                              "or load a saved model.")
        return model.module if hasattr(model, "module") else model  # type:ignore

    def get_state_dict(self) -> Dict:
        return self.state_dict

    def save(self, model_path: str, entire: bool=False) -> str:
        """
        Saves the Estimator state (including model and optimizer) or the entire model
        to the provided model_path.

        :param model_path: (str) Path to save the model.
        :param entire: (boolean) Whether to save the entire model. If False, saves the
               Estimator state. Default is False.
        :return:
        """
        if is_local_path(model_path):
            if entire:
                torch.save(self.get_model(), model_path)
            else:
                torch.save(self.state_dict, model_path)
        else:
            file_name = os.path.basename(model_path)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file_name)
            try:
                if entire:
                    torch.save(self.get_model(), temp_path)
                else:
                    torch.save(self.state_dict, temp_path)
                put_local_file_to_remote(temp_path, model_path)
            finally:
                shutil.rmtree(temp_dir)
        return model_path

    def load(self, model_path: str):
        """
        Loads the Estimator state (including model and optimizer) or the entire model
        from the provided model_path.

        :param model_path: (str) Path to the existing model. Model class must be defined
               on the driver when loading the entire model.
        """
        import torch.nn as nn
        if is_local_path(model_path):
            res = torch.load(model_path)
        else:
            file_name = os.path.basename(model_path)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file_name)
            try:
                get_remote_file_to_local(model_path, temp_path)
                res = torch.load(temp_path)
            finally:
                shutil.rmtree(temp_dir)
        if isinstance(res, Iterable) and not isinstance(res, nn.Sequential):
            if "models" in res:
                self.state_dict = res
            else:
                self.state_dict = [re.state_dict() for re in res]
        else:
            self.state_dict = res.state_dict()
        if self.model_creator is None:
            self.model_creator = lambda config: res
            self.worker_init_params["model_creator"] = self.model_creator

    def save_checkpoint(self, model_path: str):
        """
        Manually saves the Estimator state (including model and optimizer) to the provided
        model_path.
        :param model_path: (str) Path to save the model. Both local and remote path are supported.
               e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"
        :return: None
        """
        if is_local_path(model_path):
            self.save(model_path)
        else:
            self.driver_runner.load_state_dict(self.state_dict)
            self.driver_runner.save_checkpoint(filepath=model_path)

    def load_checkpoint(self, model_path: str):
        """
        Loads the Estimator state (including model and optimizer) from the provided model_path.
        :param model_path: (str) Path to the existing model. Both local and remote path are
               supported. e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"
        :return: None
        """
        if is_local_path(model_path):
            self.load(model_path)
        else:
            self.driver_runner.load_checkpoint(filepath=model_path)
            self.state_dict = self.driver_runner.get_state_dict()

    def shutdown(self):
        """
        Shutdown estimator and release resources.
        """
        if self.need_to_log_to_driver:
            stop_log_server(self.log_server_thread, self.ip, self.log_port)

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
import logging
import numbers
import torch
import numpy as np
import copy

from bigdl.orca.learn.pytorch.training_operator import TrainingOperator
from bigdl.orca.learn.pytorch.pytorch_pyspark_worker import PytorchPysparkWorker
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, make_data_creator, update_predict_xshards, \
    reload_dataloader_creator
from bigdl.orca.data import SparkXShards
from bigdl.orca import OrcaContext
from bigdl.orca.learn.base_estimator import BaseEstimator
from bigdl.orca.data.file import get_remote_file_to_local, enable_multi_fs_save, \
    enable_multi_fs_load
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.orca.learn.log_monitor import start_log_server, stop_log_server

from bigdl.orca.learn.utils import find_free_port, find_ip_and_free_port
from bigdl.dllib.utils.utils import get_node_ip
from bigdl.dllib.utils.log4Error import *


def partition_to_creator(partition):
    def data_creator(config, batch_size):
        from bigdl.orca.data.utils import ray_partition_get_data_label, index_data, get_size
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
        data, label = ray_partition_get_data_label(partition,
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
            workers_per_node=1,
            sync_stats=True,
            log_level=logging.INFO,
            model_dir=None,
            log_to_driver=True):
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
        if not (isinstance(model_creator, types.FunctionType) and
                isinstance(optimizer_creator, types.FunctionType)):  # Torch model is also callable.
            invalidInputError(False,
                              "Must provide a function for both model_creator and"
                              " optimizer_creator")

        if not training_operator_cls and not loss_creator:
            invalidInputError(False,
                              "If a loss_creator is not provided, you must "
                              "provide a custom training operator.")

        self.model_dir = parse_model_dir(model_dir)

        self.model_creator = model_creator
        self.initialization_hook = initialization_hook

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
            training_operator_cls=training_operator_cls,
            scheduler_step_freq=scheduler_step_freq,
            use_tqdm=use_tqdm,
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

        self.state_dict = self.driver_runner.get_state_dict()

    def create_tcpstore_server(self):
        import torch.distributed as dist
        server_store = dist.TCPStore(self.ip, self.tcp_store_port, -1, True,
                                     dist.constants.default_pg_timeout)
        return server_store

    def _get_cluster_info(self, sc):
        cluster_info = self.workerRDD.barrier().mapPartitions(find_ip_and_free_port).collect()
        return cluster_info

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
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param callbacks: A list for all callbacks.

        :return: A list of dictionary of metrics for every training epoch. If reduce_results is
                False, this will return a nested list of metric dictionaries whose length will be
                equal to the total number of workers.
                You can also provide custom metrics by passing in a custom training_operator_cls
                when creating the Estimator.
        """
        data, validation_data = maybe_dataframe_to_xshards(data,
                                                           validation_data=validation_data,
                                                           feature_cols=feature_cols,
                                                           label_cols=label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers)

        sc = OrcaContext.get_spark_context()
        _ = self.create_tcpstore_server()
        cluster_info = self._get_cluster_info(sc)
        state_dict = self._get_broadcasted_state_dict(sc)
        init_params = dict(
            mode="fit",
            state_dict=state_dict,
            cluster_info=cluster_info)
        init_params.update(self.worker_init_params)

        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            profile=profile,
            info=info,
            callbacks=callbacks,
        )

        if isinstance(data, SparkXShards):
            # set train/validation
            params["wrap_dataloader"] = False

            if validation_data is None:
                def transform_func(iter, init_params, param):
                    partition_data = list(iter)
                    param["data_creator"] = partition_to_creator(partition_data)
                    runner = PytorchPysparkWorker(**init_params)
                    result = runner.train_epochs(**param)
                    runner.shutdown()
                    return result

                res = data.rdd.repartition(self.num_workers).barrier() \
                    .mapPartitions(
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

                train_rdd = data.rdd.mapPartitions(lambda iter: [list(iter)])
                val_rdd = validation_data.rdd.mapPartitions(lambda iter: [list(iter)])
                res = train_rdd.zip(val_rdd).repartition(self.num_workers).barrier() \
                    .mapPartitions(
                    lambda iter: transform_func(iter, init_params, params)).collect()

        else:
            if not isinstance(data, types.FunctionType):
                invalidInputError(False,
                                  "data should be either an instance of SparkXShards or a "
                                  "callable  function, but got type: {}".format(type(data)))

            params["data_creator"] = reload_dataloader_creator(data)
            params["validation_data_creator"] = reload_dataloader_creator(validation_data)

            def transform_func(iter, init_param, param):
                return PytorchPysparkWorker(**init_param).train_epochs(**param)

            res = self.workerRDD.barrier().mapPartitions(
                lambda iter: transform_func(iter, init_params, params)).collect()

        if self.model_dir is not None:
            self.state_dict = PyTorchPySparkEstimator._get_state_dict_from_remote(self.model_dir)
            worker_stats = res
        else:
            self.state_dict = res[0]
            worker_stats = res[1]

        epoch_stats = list(map(list, zip(*worker_stats)))
        if reduce_results:
            for i in range(len(epoch_stats)):
                epoch_stats[i] = self._process_stats(epoch_stats[i])
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

        pred_shards = SparkXShards(xshards.rdd.mapPartitions(
            lambda iter: transform_func(iter, init_params, params)))
        return pred_shards

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
        from pyspark.sql import DataFrame

        sc = OrcaContext.get_spark_context()
        cluster_info = self._get_cluster_info(sc)
        state_dict = self._get_broadcasted_state_dict(sc)

        init_params = dict(
            mode="predict",
            state_dict=state_dict,
            cluster_info=cluster_info,
        )
        init_params.update(self.worker_init_params)

        params = dict(
            batch_size=batch_size,
            profile=profile
        )

        if isinstance(data, DataFrame):
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict")

            pred_shards = self._predict_spark_xshards(xshards, init_params, params)
            result = convert_predict_xshards_to_dataframe(data, pred_shards)

        elif isinstance(data, SparkXShards):
            pred_shards = self._predict_spark_xshards(data, init_params, params)
            result = update_predict_xshards(data, pred_shards)
        else:
            invalidInputError(False,
                              "Only xshards or Spark DataFrame is supported for predict")

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
        sc = OrcaContext.get_spark_context()
        cluster_info = self._get_cluster_info(sc)
        state_dict = self._get_broadcasted_state_dict(sc)
        init_params = dict(
            mode="evaluate",
            state_dict=state_dict,
            cluster_info=cluster_info)

        init_params.update(self.worker_init_params)

        params = dict(
            batch_size=batch_size,
            num_steps=num_steps,
            profile=profile,
            info=info)

        from bigdl.orca.data import SparkXShards
        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers)
        if isinstance(data, SparkXShards):
            # set train/validation data
            def transform_func(iter, init_param, param):
                partition_data = list(iter)
                param["data_creator"] = partition_to_creator(partition_data)
                return PytorchPysparkWorker(**init_param).validate(**param)

            res = data.rdd.repartition(self.num_workers).barrier() \
                .mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params["data_creator"] = reload_dataloader_creator(data)

            def transform_func(iter, init_param, param):
                return PytorchPysparkWorker(**init_param).validate(**param)

            res = self.workerRDD.barrier().mapPartitions(
                lambda iter: transform_func(iter, init_params, params)).collect()

        return self._process_stats(res)

    def get_model(self):
        """
        Returns the learned PyTorch model.

        :return: The learned PyTorch model.
        """
        state = self.state_dict
        model = self.model_creator(self.config)
        model_state = state["models"][0]
        model.load_state_dict(model_state)
        return model.module if hasattr(model, "module") else model

    def get_state_dict(self):
        return self.state_dict

    @enable_multi_fs_save
    def save(self, model_path):
        """
        Saves the Estimator state (including model and optimizer) to the provided model_path.

        :param model_path: (str) Path to save the model.
        :return:
        """
        state_dict = self.state_dict
        torch.save(state_dict, model_path)
        return model_path

    @enable_multi_fs_load
    def load(self, model_path):
        """
        Loads the Estimator state (including model and optimizer) from the provided model_path.

        :param model_path: (str) Path to the existing model.
        """
        state_dict = torch.load(model_path)
        self.state_dict = state_dict

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
            self.driver_runner.load_state_dict(self.state_dict)
            self.driver_runner.save_checkpoint(filepath=model_path)

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
            self.driver_runner.load_checkpoint(filepath=model_path)
            self.state_dict = self.driver_runner.get_state_dict()

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

    def shutdown(self):
        """
        Shutdown estimator and release resources.
        """
        if self.need_to_log_to_driver:
            stop_log_server(self.log_server_thread, self.ip, self.port)

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

import logging
import os
import threading
import json
import random

from pyspark.sql.dataframe import DataFrame
import numpy as np

from bigdl.orca.learn.utils import session_execute, decode
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils.file_utils import enable_multi_fs_load, enable_multi_fs_save

from bigdl.orca.learn.tf2.spark_runner import SparkRunner
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, make_data_creator, update_predict_xshards, \
    process_xshards_of_pandas_dataframe
from bigdl.orca.data.shard import SparkXShards
from bigdl.orca import OrcaContext

LOG_FILE_CHANNEL = "SPARK_LOG_CHANNEL"

logger = logging.getLogger(__name__)

class SparkTFEstimator():
    def __init__(self,
                 model_creator,
                 config=None,
                 compile_args_creator=None,
                 verbose=False,
                 workers_per_node=1,
                 log_to_driver=True,
                 **kwargs):
        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.verbose = verbose

        sc = OrcaContext.get_spark_context()

        num_node, num_core = get_node_and_core_number()
        self.num_workers = num_node * workers_per_node

        if not "inter_op_parallelism"  in self.config:
            self.config["inter_op_parallelism"] = 1
        if not "intra_op_parallelism" in self.config:
            self.config["intra_op_parallelism"] = num_core // workers_per_node

        self.model_weights = None

        if "batch_size" in self.config:
            raise Exception("Please do not specify batch_size in config. Input batch_size in the"
                            " fit/evaluate function of the estimator instead.")
        # start redis server
        from bigdl.dllib.utils.utils import get_node_ip
        self.ip = get_node_ip()

        if "redis_port" in kwargs:
            self.redis_port = kwargs["redis_port"]
        else:
            self.redis_port = random.randint(10000, 65535)
        if "redis_password" in kwargs:
            self.redis_password = kwargs["redis_password"]
        else:
            self.redis_password = None
        self.is_local = sc.master.startswith("local")
        if not self.is_local:
            if log_to_driver:
                # start redis
                process_info = self._start_redis()

                # print executor logs
                import redis
                redis_client = redis.StrictRedis(
                    host=self.ip, port=self.port)
                threads_stopped = threading.Event()
                logger_thread = threading.Thread(
                    target=self._print_logs,
                    name="print_logs",
                    args=(redis_client, threads_stopped))
                logger_thread.daemon = True
                logger_thread.start()


    def fit(self, data, epochs=1, batch_size=32, verbose=1,
            callbacks=None, validation_data=None, class_weight=None,
            steps_per_epoch=None, validation_steps=None, validation_freq=1,
            data_config=None, feature_cols=None,
            label_cols=None, model_dir=None):
        """
        Train this tensorflow model with train data.
        :param data: train data. It can be XShards, Spark DataFrame or creator function which
               returns Iter or DataLoader.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
        :param epochs: Number of epochs to train the model. Default: 1.
        :param batch_size: Batch size used for training. Default: 32.
        :param verbose: Prints output of one model if true.
        :param callbacks: List of Keras compatible callbacks to apply during training.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param class_weight: Optional dictionary mapping class indices (integers) to a weight
               (float) value, used for weighting the loss function. This can be useful to tell
               the model to "pay more attention" to samples from an under-represented class.
        :return:
        """
        import numpy as np
        sc = OrcaContext.get_spark_context()

        init_params = dict(
            model_creator=self.model_creator,
            compile_args_creator=self.compile_args_creator,
            config=self.config,
            verbose=self.verbose,
            size=self.num_workers,
            mode="fit",
            is_local=self.is_local,
            redis_address=":".join([self.ip, self.redis_port]),
            redis_password=self.redis_password
        )

        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            data_config=data_config
        )

        # dataframe change to xshard, num_partition >= num_workers
        data, validation_data = maybe_dataframe_to_xshards(data, validation_data,
                                                           feature_cols, label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers,
                                                           accept_str_col=True)
        if isinstance(data, SparkXShards):
            # set train/validation data
            if validation_data is None:
                def transform_func(iter, init_param, param):
                    partition_data = list(iter)
                    param["data_creator"] = make_data_creator(partition_data)
                    return SparkRunner(**init_param).step(**param)

                res = data.rdd.repartition(self.num_workers).barrier() \
                    .mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
            else:
                def transform_func(iter, init_param, param):
                    data_tuple_list = list(iter)
                    data_list = [x[0] for x in data_tuple_list]
                    valid_list = [x[1] for x in data_tuple_list]
                    param["data_creator"] = make_data_creator(data_list)
                    param["validation_data_creator"] = make_data_creator(valid_list)
                    return SparkRunner(**init_param).step(**param)

                res = data.zip(validation_data).rdd.repartition(self.num_workers).barrier() \
                    .mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params["data_creator"] = data
            params["validation_data_creator"] = validation_data

            workerRDD = sc.parallelize(list(range(self.num_workers)), self.num_workers). \
                repartition(self.num_workers)

            def transform_func(iter, init_param, param):
                return SparkRunner(**init_param).step(**param)

            res = workerRDD.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()

        self.model_weights = res[0]

        return res

    def evaluate(self, data, batch_size=32, num_steps=None, verbose=1,
                 sample_weight=None, callbacks=None, data_config=None,
                 feature_cols=None, label_cols=None):
        """
        Evaluates the model on the validation data set.
        :param data: evaluate data. It can be XShards, Spark DataFrame or creator function which
               returns Iter or DataLoader.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param batch_size: Batch size used for evaluation. Default: 32.
        :param verbose: Prints output of one model if true.
        :param callbacks: List of Keras compatible callbacks to apply during evaluation.
        :param class_weight: Optional dictionary mapping class indices (integers) to a weight
               (float) value, used for weighting the loss function. This can be useful to tell
               the model to "pay more attention" to samples from an under-represented class.
        :return: validation result
        """
        import numpy as np
        sc = OrcaContext.get_spark_context()
        logger.info("Starting validation step.")

        init_params = dict(
            model_creator=self.model_creator,
            compile_args_creator=self.compile_args_creator,
            config=self.config,
            verbose=self.verbose,
            size=self.num_workers,
            model_weights=self.model_weights,
            mode="evaluate",
            is_local=self.is_local,
            redis_address=":".join([self.ip, self.redis_port]),
            redis_password=self.redis_password
        )

        params = dict(
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=num_steps,
            callbacks=callbacks,
            data_config=data_config,
        )

        # dataframe change to xshard, num_partition >= num_workers
        data, _ = maybe_dataframe_to_xshards(data, validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers,
                                             accept_str_col=True)
        if isinstance(data, SparkXShards):
            # set train/validation data
            def transform_func(iter, init_param, param):
                partition_data = list(iter)
                param["data_creator"] = make_data_creator(partition_data)
                return SparkRunner(**init_param).validate(**param)

            res = data.rdd.repartition(self.num_workers).barrier() \
                .mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params["data_creator"] = data
            # params["model_weights"] = self.model_weights

            # worker_nums = self.worker_nums
            workerRDD = sc.parallelize(list(range(self.num_workers)), self.num_workers).repartition(self.num_workers)

            def transform_func(iter, init_param, param):
                return SparkRunner(**init_param).validate(**param)

            res = workerRDD.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()

        return res[0]

    def predict(self, data, batch_size=None, verbose=1,
                steps=None, callbacks=None, data_config=None,
                feature_cols=None):
        """
        Predict the input data
        :param data: predict input data.  It can be XShards or Spark DataFrame.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature}, where feature is a numpy array or a tuple of numpy arrays.
        :param batch_size: Batch size used for inference. Default: None.
        :param verbose: Prints output of one model if true.
        :param steps: Total number of steps (batches of samples) before declaring the prediction
               round finished. Ignored with the default value of None.
        :param callbacks: List of Keras compatible callbacks to apply during prediction.
        :param data_config: An optional dictionary that can be passed to data creator function.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame or an XShards of Pandas DataFrame. Default: None.
        :return:
        """
        logger.info("Starting predict step.")

        init_params = dict(
            model_creator=self.model_creator,
            compile_args_creator=self.compile_args_creator,
            config=self.config,
            verbose=self.verbose,
            size=self.num_workers,
            model_weights=self.model_weights,
            mode="predict",
            is_local=self.is_local,
            redis_address=":".join([self.ip, self.redis_port]),
            redis_password=self.redis_password
        )

        params = dict(
            verbose=verbose,
            batch_size=batch_size,
            steps=steps,
            callbacks=callbacks,
            data_config=data_config
        )

        if isinstance(data, DataFrame):
            data = data.repartition(self.num_workers)
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict",
                                              accept_str_col=True)

            def transform_func(iter, init_param, param):
                partition_data = list(iter)
                # res = combine_in_partition(partition_data)
                param["data_creator"] = make_data_creator(partition_data)
                return SparkRunner(**init_param).predict(**param)

            pred_shards = SparkXShards(xshards.rdd.repartition(self.num_workers) \
                                       .mapPartitions(lambda iter: transform_func(iter, init_params, params)))
            result = convert_predict_xshards_to_dataframe(data, pred_shards)
        else:
            raise ValueError("Only xshards or Spark DataFrame is supported for predict")

        return result

    @enable_multi_fs_save
    def save_weights(self, filepath, overwrite=True, save_format=None):
        """
        Saves the model at the provided path.
        :param checkpoint: (str) Path to the target checkpoint file.
        """

        # Some model might need to aggregate variables during checkpointing
        # which requires both the chief and workers to participate in the
        # allreduce communication protocol.
        # So we need to call get_state on every remote workers, otherwise
        # it might get stuck
        model = self.model_creator(self.config)
        model.set_weights(self.model_weights)
        model.save_weights(filepath, overwrite, save_format)

    @enable_multi_fs_load
    def load_weights(self, filepath, by_name=False):
        """
        Save tensorflow keras model in this estimator.

        :param filepath: keras model weights save path.
        :param by_name: Boolean, whether to load weights by name or by topological
               order. Only topological loading is supported for weight files in
               TensorFlow format.
        """
        model = self.model_creator(self.config)
        model.load_weights(filepath, by_name)
        self.model_weights = model.get_weights()

    @enable_multi_fs_save
    def save(self, filepath, overwrite=True, save_format=None):
        """
        Saves the model at the provided path.
        :param checkpoint: (str) Path to the target checkpoint file.
        """

        # Some model might need to aggregate variables during checkpointing
        # which requires both the chief and workers to participate in the
        # allreduce communication protocol.
        # So we need to call get_state on every remote workers, otherwise
        # it might get stuck
        model = self.model_creator(self.config)
        model.set_weights(self.model_weights)
        model.save(filepath, overwrite=overwrite, save_format=save_format)

    @enable_multi_fs_load
    def load(self, filepath):
        """
        Save tensorflow keras model in this estimator.

        :param filepath: keras model weights save path.
        :param by_name: Boolean, whether to load weights by name or by topological
               order. Only topological loading is supported for weight files in
               TensorFlow format.
        """
        import tensorflow as tf
        model = tf.keras.models.load_model(filepath)
        self.model_weights = model.get_weights()

    def _start_redis(self):
        import random
        self.redis_port = random.randint(10000, 65535)
        redis_exec = "redis-server"
        command = [redis_exec]
        if self.redis_password:
            command += ["--requirepass", self.redis_password]
        command += ["--port", str(self.redis_port), "--loglevel", "warning"]
        process_info = session_execute(command=command)
        return process_info

    def _print_logs(self, redis_client, threads_stopped):
        """Prints log messages from workers on all of the nodes.

        Args:
            redis_client: A client to the primary Redis shard.
            threads_stopped (threading.Event): A threading event used to signal to
                the thread that it should exit.
        """
        import redis
        pubsub_client = redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub_client.subscribe(LOG_FILE_CHANNEL)
        try:
            # Keep track of the number of consecutive log messages that have been
            # received with no break in between. If this number grows continually,
            # then the worker is probably not able to process the log messages as
            # rapidly as they are coming in.
            num_consecutive_messages_received = 0
            while True:
                # Exit if we received a signal that we should stop.
                if threads_stopped.is_set():
                    return

                msg = pubsub_client.get_message()
                if msg is None:
                    num_consecutive_messages_received = 0
                    threads_stopped.wait(timeout=0.01)
                    continue
                num_consecutive_messages_received += 1

                data = json.loads(decode(msg["data"]))
                for line in data["lines"]:
                    print("(executor {}, ip={}) {}".format(
                            data["executor_id"],
                            data["ip"], line))

        except (OSError, redis.exceptions.ConnectionError) as e:
            logger.error("print_logs: {}".format(e))
        finally:
            # Close the pubsub client to avoid leaking file descriptors.
            pubsub_client.close()




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
import json
import logging
import os
import tempfile
import shutil

import tensorflow as tf
from contextlib import closing
import socket

from pyspark import BarrierTaskContext, TaskContext

from bigdl.orca.data.utils import ray_partition_get_data_label
from bigdl.orca.data.file import put_local_dir_to_remote
from bigdl.orca.learn.utils import save_pkl, duplicate_stdout_stderr_to_file,\
    get_specific_object_from_callbacks, get_replaced_path
from bigdl.orca.learn.log_monitor import LogMonitor

logger = logging.getLogger(__name__)


def find_free_port(tc):
    address = tc.getTaskInfos()[tc.partitionId()].address.split(":")[0]
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tc.barrier()
        return f"{address}:{s.getsockname()[1]}"


def handle_datasets_train(data_creator, validation_data_creator):
        train_dataset = data_creator()
        if validation_data_creator is not None:
            test_dataset = validation_data_creator()
        else:
            test_dataset = None
        return train_dataset, test_dataset


class DatasetHandler:

    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def handle_datasets_train(self, data_creator,
                              validation_data_creator,
                              config, epochs, steps_per_epoch,
                              validation_steps):

        config, local_batch_size = self._handle_batch_size(config)
        config['rank'] = self.rank
        config['size'] = self.size
        train_dataset = data_creator(config, config["batch_size"])
        if isinstance(train_dataset, list) and \
           all([isinstance(x, dict) for x in train_dataset]):
            assert steps_per_epoch is not None, "steps_per_epoch must be provided for xshard"
            train_dataset = self._handle_xshards(train_dataset,
                                                 steps=steps_per_epoch * epochs,
                                                 local_batch_size=local_batch_size,
                                                 shuffle=True)
        else:
            train_dataset = self._handle_sharding(train_dataset)

        if validation_data_creator is not None:
            test_dataset = validation_data_creator(config, config["batch_size"])
            if isinstance(test_dataset, list) and \
                    all([isinstance(x, dict) for x in test_dataset]):
                assert validation_steps is not None, "validation_steps must be provided" \
                                                     "when use xshards for evaluate"
                test_dataset = self._handle_xshards(test_dataset,
                                                    steps=validation_steps,
                                                    local_batch_size=local_batch_size,
                                                    shuffle=False)
            else:
                test_dataset = self._handle_sharding(test_dataset)
        else:
            test_dataset = None

        return train_dataset, test_dataset

    def handle_dataset_validation(self, data_creator, config, steps):
        config, local_batch_size = self._handle_batch_size(config)
        config['rank'] = self.rank
        config['size'] = self.size
        dataset = data_creator(config, config["batch_size"])
        if isinstance(dataset, list) and all([isinstance(x, dict) for x in dataset]):
            assert steps is not None, "steps must be provided for xshard"
            dataset = self._handle_xshards(dataset,
                                           steps=steps,
                                           local_batch_size=local_batch_size,
                                           shuffle=False)
        else:
            dataset = self._handle_sharding(dataset)

        return dataset

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        raise NotImplementedError

    def _handle_sharding(self, dataset):
        raise NotImplementedError

    def _handle_batch_size(self, config):
        raise NotImplementedError

    @staticmethod
    def get_handler(backend, rank, size):

        if backend == "tf-distributed":
            return TFDistributedDatasetHandler(rank, size)

        if backend == "tf-local":
            return LocalDatasetHandler(rank, size)

        raise Exception(f"invalid backend: {backend}")


class TFDistributedDatasetHandler(DatasetHandler):

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        import tensorflow as tf

        data, label = ray_partition_get_data_label(dataset,
                                                   allow_tuple=True,
                                                   allow_list=False)

        def dataset_fn(input_context):
            dataset = tf.data.Dataset.from_tensor_slices((data, label))
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = \
                tf.data.experimental.AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            dataset = dataset.repeat()
            dataset = dataset.take(steps * local_batch_size)
            if shuffle:
                dataset = dataset.shuffle(local_batch_size * min(steps, 10))
            dataset = dataset.batch(local_batch_size)
            return dataset

        from tensorflow.python.distribute import distribution_strategy_context as ds_context
        strategy = ds_context.get_strategy()
        dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn)
        return dataset

    def _handle_sharding(self, dataset):
        return dataset

    def _handle_batch_size(self, config):
        assert "batch_size" in config, "batch_size must be set in config"
        local_batch_size = config["batch_size"] // self.size
        return config, local_batch_size


class LocalDatasetHandler(DatasetHandler):

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        import tensorflow as tf
        data, label = ray_partition_get_data_label(dataset,
                                                   allow_tuple=True,
                                                   allow_list=False)
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.repeat()
        dataset = dataset.take(steps * local_batch_size)
        if shuffle:
            dataset = dataset.shuffle(local_batch_size * min(steps, 10))
        dataset = dataset.batch(local_batch_size)
        return dataset

    def _handle_sharding(self, dataset):
        return dataset

    def _handle_batch_size(self, config):
        assert "batch_size" in config, "batch_size must be set in config"
        return config, config["batch_size"]


def find_ip_and_port(pre_iter):
    tc = BarrierTaskContext().get()
    free_port = find_free_port(tc)
    return [free_port]


class SparkRunner:
    def __init__(self, model_creator, compile_args_creator,
                 size,
                 cluster_info,
                 config=None,
                 verbose=False,
                 model_weights=None,
                 backend="tf-distributed",
                 mode="fit",
                 model_dir=None,
                 epoch=0,
                 need_to_log_to_driver=False,
                 driver_ip=None,
                 driver_port=None
                 ):
        """Initializes the runner.
                Args:
                    model_creator (dict -> Model): see tf_trainer.py.
                    data_creator (dict -> tf.Dataset, tf.Dataset): see tf_trainer.py.
                    config (dict): see tf_trainer.py.
                    verbose (bool): Outputs training data if true.
                """

        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.inter_op_parallelism = self.config.get("inter_op_parallelism", 1)
        self.intra_op_parallelism = self.config.get("intra_op_parallelism", 1)
        self.epoch = epoch
        self.verbose = verbose
        self.model_weights = model_weights
        self.size = size
        self.mode = mode
        self.backend = backend
        self.setup()
        self.cluster = cluster_info
        if TaskContext.get():
            self.partition_id = TaskContext.get().partitionId()
        else:
            self.partition_id = BarrierTaskContext.get().partitionId()
        self.need_to_log_to_driver = need_to_log_to_driver
        if need_to_log_to_driver:
            self.log_path = os.path.join(tempfile.gettempdir(),
                                         "{}_runner.log".format(self.partition_id))
            duplicate_stdout_stderr_to_file(self.log_path)
            self.logger_thread, self.thread_stop = \
                LogMonitor.start_log_monitor(driver_ip=driver_ip,
                                             driver_port=driver_port,
                                             log_path=self.log_path,
                                             partition_id=self.partition_id)

        if self.backend == "tf-distributed":
            if mode == "fit" or mode == "evaluate":
                self.setup_distributed(self.cluster)
        self.model_dir = model_dir

    def setup(self):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_parallelism)
        os.environ["KMP_BLOCKING_TIME"] = self.config.get("KMP_BLOCKING_TIME",
                                                          os.environ.get("KMP_BLOCKING_TIME", "0"))

    def _get_rank(self, cluster_info):
        # As task placement may not be identical between two different jobs,
        # we cannot simply index cluster_info using partitionId to get current
        # ip and port.
        # The approach here is to first get all tasks' ip in this job and compute
        # a local rank by counting how many tasks has the same ip but with lower id.
        # We then use the local rank to find the right slot in cluster_info to find
        # the right global_rank.
        tc = BarrierTaskContext().get()
        infos = tc.getTaskInfos()
        idx = tc.partitionId()
        local_ip = infos[idx].address.split(":")[0]
        local_rank = 0
        for i in range(0, idx):
            if infos[i].address.startswith(local_ip):
                local_rank += 1
        global_rank = -1
        local_count = 0
        for node in cluster_info:
            if node.startswith(local_ip):
                local_count += 1
            global_rank += 1
            if local_count == local_rank + 1:
                break
        return global_rank

    def setup_distributed(self, cluster):
        """
        Sets up TensorFLow distributed environment and initializes the model.
        """
        self.rank = self._get_rank(cluster)
        logger.info("cluster is: {}".format(cluster))

        os.environ["TF_CONFIG"] = json.dumps({
            'cluster': {
                'worker': cluster
            },
            'task': {'type': 'worker', 'index': self.rank}
        })
        ips = set([node.split(":")[0] for node in cluster])
        os.environ["no_proxy"] = ",".join(ips)

        self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

        # For use in model.evaluate()
        self.local_model = None

    def distributed_train_func(self, data_creator, config, epochs=1, verbose=1,
                               callbacks=None, initial_epoch=0, validation_data_creator=None,
                               class_weight=None, steps_per_epoch=None, validation_steps=None,
                               validation_freq=1):
        """
        Sets up TensorFLow distributed environment, initializes the model,
        runs a training epoch and updates the model parameters
        """
        with self.strategy.scope():
            model = self.model_creator(self.config)
            if self.model_weights:
                model.set_weights(self.model_weights.value)

            dataset_handler = DatasetHandler.get_handler(self.backend, self.rank, self.size)
            train_dataset, test_dataset = dataset_handler \
                .handle_datasets_train(data_creator=data_creator,
                                       validation_data_creator=validation_data_creator,
                                       config=config, epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_steps=validation_steps)
        checkpoint = None
        if callbacks:
            checkpoint = get_specific_object_from_callbacks(tf.keras.callbacks.ModelCheckpoint,
                                                            callbacks)
            if checkpoint:
                original_checkpoint_dir = os.path.dirname(checkpoint.filepath)
                replaced_checkpoint_path = get_replaced_path(checkpoint.filepath)
                checkpoint.filepath = replaced_checkpoint_path

        history = model.fit(train_dataset,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=callbacks,
                            validation_data=test_dataset,
                            class_weight=class_weight,
                            initial_epoch=initial_epoch,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            validation_freq=validation_freq)

        if checkpoint:
            try:
                if self.rank == 0:
                    put_local_dir_to_remote(os.path.dirname(replaced_checkpoint_path),
                                            original_checkpoint_dir)
            finally:
                shutil.rmtree(os.path.dirname(replaced_checkpoint_path))

        return (model, history)

    def step(self, data_creator, epochs=1, batch_size=32, verbose=1,
             callbacks=None, validation_data_creator=None, class_weight=None,
             steps_per_epoch=None, validation_steps=None, validation_freq=1,
             data_config=None):
        """
        Get model training results and new model.
        """
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)
        config["batch_size"] = batch_size
        val_data_creator = validation_data_creator

        model, history = self.distributed_train_func(data_creator,
                                                     config,
                                                     epochs=self.epoch + epochs,
                                                     verbose=verbose,
                                                     callbacks=callbacks,
                                                     steps_per_epoch=steps_per_epoch,
                                                     class_weight=class_weight,
                                                     initial_epoch=self.epoch,
                                                     validation_data_creator=val_data_creator,
                                                     validation_steps=validation_steps,
                                                     validation_freq=validation_freq
                                                     )
        self.epoch += epochs
        weights = model.get_weights()
        if history is None:
            stats = {}
        else:
            stats = {k: v[-1] for k, v in history.history.items()}
        if self.rank == 0:
            if self.model_dir is not None:
                model_state = {
                    "epoch": self.epoch,
                    "weights": weights,
                    "optimizer_weights": model.optimizer.get_weights()
                }
                save_pkl(model_state, os.path.join(self.model_dir, "state.pkl"))
            if self.need_to_log_to_driver:
                LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
            return [stats]
        else:
            if self.need_to_log_to_driver:
                LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
            return []

    def validate(self, data_creator, batch_size=32, verbose=1, sample_weight=None,
                 steps=None, callbacks=None, data_config=None):
        """
        Evaluates the model on the validation data set.
        """
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)
        config["batch_size"] = batch_size

        with self.strategy.scope():
            model = self.model_creator(self.config)
            if self.model_weights:
                model.set_weights(self.model_weights.value)

        with self.strategy.scope():
            dataset_handler = DatasetHandler.get_handler(self.backend,
                                                         self.rank,
                                                         self.size)

            dataset = dataset_handler.handle_dataset_validation(data_creator,
                                                                config=config,
                                                                steps=steps)

        params = dict(
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks,
        )
        results = model.evaluate(dataset, **params)

        if results is None:
            local_model = self.model_creator(self.config)
            if self.model_weights:
                local_model = local_model.set_weights(self.model_weights.value)
            results = local_model.evaluate(dataset, **params)

        if isinstance(results, list):
            stats = {
                "validation_" + k: v
                for k, v in zip(model.metrics_names, results)
            }
        else:
            stats = {"results": results}

        if self.rank == 0:
            if self.need_to_log_to_driver:
                LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
            return [stats]
        else:
            if self.need_to_log_to_driver:
                LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
            return []

    def predict(self, data_creator, batch_size, verbose, steps, callbacks, data_config):
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)

        dataset = data_creator(config, batch_size)
        partition = dataset
        params = dict(
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
        )

        if self.backend == "tf-distributed":
            local_model = self.model_creator(self.config)
            if self.model_weights:
                local_model.set_weights(self.model_weights.value)
        else:
            local_model = self.model_creator(self.config)

        def predict_fn(shard):
            y = local_model.predict(shard["x"], **params)
            return {"prediction": y}

        new_part = [predict_fn(shard) for shard in partition]

        if self.need_to_log_to_driver:
            LogMonitor.stop_log_monitor(self.log_path, self.logger_thread, self.thread_stop)
        return new_part

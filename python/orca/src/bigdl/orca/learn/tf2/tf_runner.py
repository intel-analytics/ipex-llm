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

# Copyright 2017 The Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import os

import numpy as np

import ray
from contextlib import closing
import logging
import socket

from bigdl.orca.data.utils import ray_partitions_get_data_label

logger = logging.getLogger(__name__)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _try_import_strategy():
    """Late import for Tesnorflow"""
    import tensorflow as tf
    return tf.distribute.experimental.MultiWorkerMirroredStrategy


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
                all([isinstance(x, ray.ObjectID) for x in train_dataset]):
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
                    all([isinstance(x, ray.ObjectID) for x in test_dataset]):
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
        if isinstance(dataset, list) and all([isinstance(x, ray.ObjectID) for x in dataset]):
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

        if backend == "horovod":
            return HorovodDatasetHanlder(rank, size)

        if backend == "tf-distributed":
            return TFDistributedDatasetHandler(rank, size)

        if backend == "tf-local":
            return LocalDatasetHandler(rank, size)

        raise Exception(f"invalid backend: {backend}")


class HorovodDatasetHanlder(DatasetHandler):

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        import tensorflow as tf
        data, label = ray_partitions_get_data_label(ray.get(dataset),
                                                    allow_tuple=True,
                                                    allow_list=False)
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        dataset = dataset.repeat()
        dataset = dataset.take(steps * local_batch_size)
        if shuffle:
            dataset = dataset.shuffle(local_batch_size * min(steps, 10))
        dataset = dataset.batch(local_batch_size)
        return dataset

    def _handle_sharding(self, dataset):
        from tensorflow.python.distribute.input_ops import auto_shard_dataset
        dataset = auto_shard_dataset(dataset, self.size, self.rank)
        return dataset

    def _handle_batch_size(self, config):
        assert "batch_size" in config, "batch_size must be set in config"
        config["batch_size"] = config["batch_size"] // self.size
        return config, config["batch_size"]


class TFDistributedDatasetHandler(DatasetHandler):

    def _handle_xshards(self, dataset, steps, local_batch_size, shuffle):
        import tensorflow as tf

        data, label = ray_partitions_get_data_label(ray.get(dataset),
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
        data, label = ray_partitions_get_data_label(ray.get(dataset),
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


class TFRunner:
    """Manages a TensorFlow model for training."""

    def __init__(self, model_creator, compile_args_creator,
                 config=None,
                 verbose=False):
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
        self.epoch = 0
        self.verbose = verbose

    def setup(self):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_parallelism)
        os.environ["KMP_BLOCKING_TIME"] = self.config.get("KMP_BLOCKING_TIME",
                                                          os.environ.get("KMP_BLOCKING_TIME", "0"))

    def setup_local(self):
        """Initializes the model."""
        logger.debug("Creating model")
        self.model = self.model_creator(self.config)
        self.model.compile(**self.compile_args_creator(self.config))
        self.backend = "tf-local"
        self.size = 1
        self.rank = 0
        from tensorflow.python.distribute import distribution_strategy_context as ds_context
        self.strategy = ds_context.get_strategy()

    def setup_horovod(self):
        import horovod.tensorflow.keras as hvd
        hvd.init()
        self.model = self.model_creator(self.config)
        compile_args = self.compile_args_creator(self.config)
        compile_args["optimizer"] = hvd.DistributedOptimizer(compile_args["optimizer"])

        self.model.compile(**compile_args)
        self.backend = "horovod"
        self.size = hvd.size()
        self.rank = hvd.rank()
        from tensorflow.python.distribute import distribution_strategy_context as ds_context
        self.strategy = ds_context.get_strategy()

    def setup_distributed(self, urls, world_rank, world_size):
        """Sets up TensorFLow distributed environment and initializes the model.
        Args:
            urls (str): the URLs that each node uses to connect.
            world_rank (int): the index of the runner.
            world_size (int): the total number of runners.
        """
        assert len(urls) == world_size
        tf_config = {
            "cluster": {
                "worker": urls
            },
            "task": {
                "index": world_rank,
                "type": "worker"
            }
        }
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        no_proxy = os.environ.get("no_proxy", "")
        ips = [url.split(":")[0] for url in urls]
        os.environ["no_proxy"] = ",".join(ips) + "," + no_proxy

        MultiWorkerMirroredStrategy = _try_import_strategy()

        # MultiWorkerMirroredStrategy handles everything for us, from
        # sharding the dataset (or even sharding the data itself if the loader
        # reads files from disk) to merging the metrics and weight updates
        #
        # worker 0 is the "chief" worker and will handle the map-reduce
        # every worker ends up with the exact same metrics and model
        # after model.fit
        #
        # because of this, we only really ever need to query its state
        self.strategy = MultiWorkerMirroredStrategy()

        logger.debug("Creating model with MultiWorkerMirroredStrategy")
        with self.strategy.scope():
            self.model = self.model_creator(self.config)

        # For use in model.evaluate()
        self.local_model = None
        self.backend = "tf-distributed"
        self.size = world_size
        self.rank = world_rank

    def step(self, data_creator, epochs=1, batch_size=32, verbose=1,
             callbacks=None, validation_data_creator=None, class_weight=None,
             steps_per_epoch=None, validation_steps=None, validation_freq=1,
             data_config=None):
        """Runs a training epoch and updates the model parameters."""
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)
        config["batch_size"] = batch_size

        with self.strategy.scope():
            dataset_handler = DatasetHandler.get_handler(self.backend, self.rank, self.size)
            train_dataset, test_dataset = dataset_handler \
                .handle_datasets_train(data_creator,
                                       validation_data_creator,
                                       config=config, epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_steps=validation_steps)
        # process other arguments
        if self.backend == "horovod":
            import horovod.tensorflow.keras as hvd
            hvd_callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                             hvd.callbacks.MetricAverageCallback()]
            if hvd.rank() != 0:
                verbose = 0

            if callbacks is not None:
                callbacks = hvd_callbacks + callbacks
            else:
                callbacks = hvd_callbacks
        elif self.backend == "tf-distributed":
            if self.strategy.cluster_resolver.task_id != 0:
                verbose = 0

        history = self.model.fit(train_dataset,
                                 epochs=self.epoch + epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 validation_data=test_dataset,
                                 class_weight=class_weight,
                                 initial_epoch=self.epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps,
                                 validation_freq=validation_freq)
        if history is None:
            stats = {}
        else:
            stats = {"train_" + k: v[-1] for k, v in history.history.items()}

        self.epoch += epochs
        return [stats]

    def validate(self, data_creator, batch_size=32, verbose=1, sample_weight=None,
                 steps=None, callbacks=None, data_config=None):
        """Evaluates the model on the validation data set."""
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)
        config["batch_size"] = batch_size

        with self.strategy.scope():
            dataset_handler = DatasetHandler.get_handler(self.backend,
                                                         self.rank,
                                                         self.size)

            dataset = dataset_handler.handle_dataset_validation(data_creator,
                                                                config=config,
                                                                steps=steps)

        if self.backend == "horovod":
            import horovod.tensorflow.keras as hvd
            if hvd.rank() != 0:
                verbose = 0
        elif self.backend == "tf-distributed":
            if self.strategy.cluster_resolver.task_id != 0:
                verbose = 0

        params = dict(
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks,
        )
        results = self.model.evaluate(dataset, **params)
        if results is None:
            # Using local Model since model.evaluate() returns None
            # for MultiWorkerMirroredStrategy
            logger.warning("Running a local model to get validation score.")
            self.local_model = self.model_creator(self.config)
            self.local_model.set_weights(self.model.get_weights())
            results = self.local_model.evaluate(dataset, **params)

        if isinstance(results, list):
            stats = {
                "validation_" + k: v
                for k, v in zip(self.model.metrics_names, results)
            }
        else:
            stats = {"results": results}

        return [stats]

    def predict(self, data_creator, batch_size, verbose, steps, callbacks, data_config):
        config = self.config.copy()
        if data_config is not None:
            config.update(data_config)

        dataset = data_creator(config, batch_size)
        if not isinstance(dataset, ray.ObjectID):
            raise ValueError("Only xshards is supported for predict")

        partition = ray.get(dataset)
        params = dict(
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
        )

        if self.backend == "tf-distributed":
            local_model = self.model_creator(self.config)
            local_model.set_weights(self.model.get_weights())
        else:
            local_model = self.model

        def predict_fn(shard):
            y = local_model.predict(shard["x"], **params)
            return {"prediction": y}

        new_part = [predict_fn(shard) for shard in partition]

        return new_part

    def get_state(self):
        """Returns the state of the runner."""
        return {
            "epoch": self.epoch,
            "weights": self.model.get_weights(),
            "optimizer_weights": self.model.optimizer.get_weights()
        }

    def set_state(self, state):
        """Sets the state of the model."""
        self.epoch = state["epoch"]
        self.model.set_weights(state["weights"])

    def shutdown(self):
        """Attempts to shut down the worker."""
        del self.model

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray._private.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return find_free_port()

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

import ray
import ray.services
from contextlib import closing
import logging
import socket
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
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_parallelism)
        os.environ["OMP_NUM_THREADS"] = self.config.get("OMP_NUM_THREADS",
                                                        str(self.intra_op_parallelism))
        os.environ["KMP_BLOCKING_TIME"] = self.config.get("KMP_BLOCKING_TIME",
                                                          os.environ.get("KMP_BLOCKING_TIME", "0"))

        self.epoch = 0
        self.verbose = verbose

    def setup(self):
        """Initializes the model."""
        logger.debug("Creating model")
        self.model = self.model_creator(self.config)
        self.model.compile(**self.compile_args_creator(self.config))
        self.backend = "tf-local"

    def setup_horovod(self):
        import horovod.tensorflow.keras as hvd
        hvd.init()
        self.model = self.model_creator(self.config)
        compile_args = self.compile_args_creator(self.config)
        compile_args["optimizer"] = hvd.DistributedOptimizer(compile_args["optimizer"])

        self.model.compile(**compile_args)
        self.backend = "horovod"

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
            self.model.compile(**self.compile_args_creator(self.config))

        # For use in model.evaluate()
        self.local_model = None
        self.backend = "tf-distributed"

    def step(self, data_creator, epochs=1, verbose=1,
             callbacks=None, validation_data_creator=None, class_weight=None,
             steps_per_epoch=None, validation_steps=None, validation_freq=1):
        """Runs a training epoch and updates the model parameters."""

        train_dataset = data_creator(self.config)
        if validation_data_creator is not None:
            test_dataset = validation_data_creator(self.config)
        else:
            test_dataset = None

        if self.backend == "horovod":
            import horovod.tensorflow.keras as hvd
            train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
            if test_dataset is not None:
                test_dataset = test_dataset.shard(hvd.size(), hvd.rank())

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
        return stats

    def validate(self, data_creator, verbose=1, sample_weight=None,
                 steps=None, callbacks=None):
        """Evaluates the model on the validation data set."""

        dataset = data_creator(self.config)

        if self.backend == "horovod":
            import horovod.tensorflow.keras as hvd
            dataset = dataset.shard(hvd.size(), hvd.rank())

        if self.backend == "horovod":
            import horovod.tensorflow.keras as hvd
            if hvd.rank() != 0:
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

        return stats

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
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return find_free_port()

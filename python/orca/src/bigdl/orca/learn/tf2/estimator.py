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
import tempfile
import shutil
import os
import numpy as np

import tensorflow as tf
from bigdl.dllib.utils.file_utils import is_local_path, get_remote_dir_to_local


logger = logging.getLogger(__name__)


class Estimator(object):
    @staticmethod
    def from_keras(*,
                   model_creator,
                   config=None,
                   verbose=False,
                   workers_per_node=1,
                   compile_args_creator=None,
                   backend="tf2",
                   cpu_binding=False,
                   log_to_driver=True,
                   model_dir=None,
                   **kwargs
                   ):
        """
        Create an Estimator for tensorflow 2.

        :param model_creator: (dict -> Model) This function takes in the `config`
               dict and returns a compiled TF model.
        :param config: (dict) configuration passed to 'model_creator',
               'data_creator'. Also contains `fit_config`, which is passed
               into `model.fit(data, **fit_config)` and
               `evaluate_config` which is passed into `model.evaluate`.
        :param verbose: (bool) Prints output of one model if true.
        :param workers_per_node: (Int) worker number on each node. default: 1.
        :param compile_args_creator: (dict -> dict of loss, optimizer and metrics) Only used when
               the backend="horovod". This function takes in the `config` dict and returns a
               dictionary like {"optimizer": tf.keras.optimizers.SGD(lr), "loss":
               "mean_squared_error", "metrics": ["mean_squared_error"]}
        :param backend: (string) You can choose "horovod", "tf2" or "spark" as backend.
         Default: `tf2`.
        :param cpu_binding: (bool) Whether to binds threads to specific CPUs. Default: False
        :param log_to_driver: (bool) Whether display executor log on driver in cluster mode.
         Default: True. This option is only for "spark" backend.
        :param model_dir: (str) The directory to save model states. It is required for "spark"
        backend. For cluster mode, it should be a share filesystem path which can be accessed
        by executors.
        """
        if backend in {"tf2", "horovod"}:
            from bigdl.orca.learn.tf2.ray_estimator import TensorFlow2Estimator
            return TensorFlow2Estimator(model_creator=model_creator, config=config,
                                        verbose=verbose, workers_per_node=workers_per_node,
                                        backend=backend, compile_args_creator=compile_args_creator,
                                        cpu_binding=cpu_binding)
        elif backend == "spark":
            if cpu_binding:
                raise ValueError("cpu_binding should not be True when using spark backend")
            if not model_dir:
                raise ValueError("Please specify model directory when using spark backend")
            from bigdl.orca.learn.tf2.pyspark_estimator import SparkTFEstimator
            return SparkTFEstimator(model_creator=model_creator,
                                    config=config, verbose=verbose,
                                    compile_args_creator=compile_args_creator,
                                    workers_per_node=workers_per_node,
                                    log_to_driver=log_to_driver,
                                    model_dir=model_dir,
                                    **kwargs)
        else:
            raise ValueError("Only horovod, tf2 and spark backends are supported"
                             f" for now, got backend: {backend}")

    @staticmethod
    def latest_checkpoint(checkpoint_dir):
        if is_local_path(checkpoint_dir):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
            return checkpoint_path
        else:
            try:
                temp_dir = tempfile.mkdtemp()
                get_remote_dir_to_local(checkpoint_dir, temp_dir)
                checkpoint_path = tf.train.latest_checkpoint(temp_dir)
                checkpoint_prefix = os.path.basename(checkpoint_path)
                return os.path.join(checkpoint_dir, checkpoint_prefix)
            finally:
                shutil.rmtree(temp_dir)


def make_data_creator(refs):
    def data_creator(config, batch_size):
        return refs

    return data_creator


def data_length(data):
    x = data["x"]
    if isinstance(x, np.ndarray):
        return x.shape[0]
    else:
        return x[0].shape[0]

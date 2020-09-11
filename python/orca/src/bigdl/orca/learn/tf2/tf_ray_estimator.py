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

import logging
import pickle

import numpy as np
import ray

from zoo.orca.learn.tf2.tf_runner import TFRunner
from zoo.ray import RayContext

logger = logging.getLogger(__name__)


class Estimator:
    def __init__(self,
                 model_creator,
                 compile_args_creator=None,
                 config=None,
                 verbose=False,
                 backend="tf",
                 workers_per_node=1):
        """Sets up the TensorFlow trainer.

        Args:
            model_creator (dict -> Model): This function takes in the `config`
                dict and returns a compiled TF model.
            data_creator (dict -> tf.Dataset, tf.Dataset): Creates
                the training and validation data sets using the config.
                `config` dict is passed into the function.
            config (dict): configuration passed to 'model_creator',
                'data_creator'. Also contains `fit_config`, which is passed
                into `model.fit(data, **fit_config)` and
                `evaluate_config` which is passed into `model.evaluate`.
            num_replicas (int): Sets number of workers used in distributed
                training. Workers will be placed arbitrarily across the
                cluster.
            use_gpu (bool): Enables all workers to use GPU.
            verbose (bool): Prints output of one model if true.
        """
        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.verbose = verbose

        ray_ctx = RayContext.get()
        if "inter_op_parallelism" not in self.config:
            self.config["inter_op_parallelism"] = 1

        if "intra_op_parallelism" not in config:
            self.config["intra_op_parallelism"] = ray_ctx.ray_node_cpu_cores // workers_per_node

        if backend == "horovod":
            assert compile_args_creator is not None, "compile_args_creator should not be None," \
                                                     " when backend is set to horovod"

        params = {
            "model_creator": model_creator,
            "compile_args_creator": compile_args_creator,
            "config": self.config,
            "verbose": self.verbose,
        }

        if backend == "tf":
            cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            num_nodes = ray_ctx.num_ray_nodes * workers_per_node

            worker_class = ray.remote(num_cpus=cores_per_node)(TFRunner)
            self.remote_workers = [worker_class.remote(**params)
                                   for i in range(0, num_nodes)]
            ips = ray.get(
                [worker.get_node_ip.remote() for worker in self.remote_workers])
            ports = ray.get(
                [worker.find_free_port.remote() for worker in self.remote_workers])

            urls = ["{ip}:{port}".format(ip=ips[i], port=ports[i])
                    for i in range(len(self.remote_workers))]

            # Get setup tasks in order to throw errors on failure
            ray.get([
                worker.setup_distributed.remote(urls, i, len(self.remote_workers))
                for i, worker in enumerate(self.remote_workers)])
        elif backend == "horovod":
            # it is necessary to call self.run first to set horovod environment
            from zoo.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
            horovod_runner = HorovodRayRunner(ray_ctx,
                                              worker_cls=TFRunner,
                                              worker_param=params,
                                              workers_per_node=workers_per_node)
            horovod_runner.run(lambda: print("worker initialized"))
            self.remote_workers = horovod_runner.remote_workers
            ray.get([
                worker.setup_horovod.remote()
                for i, worker in enumerate(self.remote_workers)])
        else:
            raise Exception("Only \"tf\" and \"horovod\" are legal "
                            "value of backend, but got {}".format(backend))

    @classmethod
    def from_keras(cls, model_creator,
                   config=None,
                   verbose=False,
                   workers_per_node=1,
                   compile_args_creator=None,
                   backend="tf"):
        return cls(model_creator, config=config,
                   verbose=verbose, workers_per_node=workers_per_node,
                   backend=backend, compile_args_creator=compile_args_creator)

    def fit(self, data_creator, epochs=1, verbose=1,
            callbacks=None, validation_data_creator=None, class_weight=None,
            steps_per_epoch=None, validation_steps=None, validation_freq=1):
        """Runs a training epoch."""

        params = dict(
            data_creator=data_creator,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data_creator=validation_data_creator,
            class_weight=class_weight,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
        )
        worker_stats = ray.get([w.step.remote(**params) for w in self.remote_workers])
        stats = worker_stats[0].copy()
        return stats

    def evaluate(self, data_creator, verbose=1, sample_weight=None,
                 steps=None, callbacks=None):
        """Evaluates the model on the validation data set."""
        logger.info("Starting validation step.")
        params = dict(
            data_creator=data_creator,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks
        )
        # see ./tf_runner.py:setup_distributed
        # for an explanation of only taking the first worker's data
        stats = ray.get([w.validate.remote(**params) for w in self.remote_workers])
        stats = stats[0].copy()
        return stats

    def get_model(self):
        """Returns the learned model."""
        state = ray.get(self.remote_workers[0].get_state.remote())
        return self._get_model_from_state(state)

    def save(self, checkpoint):
        """Saves the model at the provided checkpoint.

        Args:
            checkpoint (str): Path to target checkpoint file.

        """

        state = ray.get(self.remote_workers[0].get_state.remote())

        with open(checkpoint, "wb") as f:
            pickle.dump(state, f)

        return checkpoint

    def restore(self, checkpoint):
        """Restores the model from the provided checkpoint.

        Args:
            checkpoint (str): Path to target checkpoint file.

        """
        with open(checkpoint, "rb") as f:
            state = pickle.load(f)

        state_id = ray.put(state)
        ray.get([worker.set_state.remote(state_id) for worker in self.remote_workers])

    def shutdown(self):
        """Shuts down workers and releases resources."""
        for worker in self.remote_workers:
            worker.shutdown.remote()
            worker.__ray_terminate__.remote()

    def _get_model_from_state(self, state):
        """Creates model and load weights from state"""

        model = self.model_creator(self.config)
        model.set_weights(state["weights"])

        # This part is due to ray.get() changing scalar np.int64 object to int
        state["optimizer_weights"][0] = np.array(
            state["optimizer_weights"][0], dtype=np.int64)

        if model.optimizer.weights == []:
            model._make_train_function()
        model.optimizer.set_weights(state["optimizer_weights"])

        return model

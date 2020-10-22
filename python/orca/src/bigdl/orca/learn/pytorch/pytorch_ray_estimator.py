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
import torch
import torch.nn as nn
import collections
import numpy as np
import numbers
import io

from zoo.orca.learn.pytorch import utils
from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.learn.pytorch.torch_runner import TorchRunner
from zoo.ray import RayContext
import ray

logger = logging.getLogger(__name__)

from ray.exceptions import RayActorError


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


class PyTorchRayEstimator:
    def __init__(
            self,
            *,
            model_creator,
            optimizer_creator,
            loss_creator=None,
            scheduler_creator=None,
            training_operator_cls=TrainingOperator,
            initialization_hook=None,
            config=None,
            scheduler_step_freq="batch",
            use_tqdm=False,
            backend="pytorch",
            workers_per_node=1):

        # todo remove ray_ctx to run on workers
        ray_ctx = RayContext.get()
        if not (callable(model_creator) and callable(optimizer_creator)):
            raise ValueError(
                "Must provide a callable model_creator and optimizer_creator")

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
            config=worker_config)

        if backend == "pytorch":
            cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            num_nodes = ray_ctx.num_ray_nodes * workers_per_node
            RemoteRunner = ray.remote(num_cpus=1)(TorchRunner)
            self.remote_workers = [
                RemoteRunner.remote(**params) for i in range(num_nodes)
            ]
            ray.get([
                worker.setup.remote(cores_per_node)
                for i, worker in enumerate(self.remote_workers)
            ])

            ip = ray.services.get_node_ip_address()
            port = utils.find_free_port()
            address = "tcp://{ip}:{port}".format(ip=ip, port=port)

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
            raise Exception("Only \"pytorch\" and \"horovod\" are legal "
                            "values of backend, but got {}".format(backend))

    def train(self,
              data_creator,
              epochs=1,
              profile=False,
              reduce_results=True,
              info=None):
        """Runs a training epoch.

        Calls `operator.train_epoch()` on N parallel workers simultaneously
        underneath the hood.
        :param data_creator: (callable) a funtion that takes a config dict as input
                  and return a data loader containing the training data.
        :param epochs: (int) Number of epochs to train the model
        :param profile: (bool) Returns time stats for the training procedure.
        :param reduce_results: (bool) Whether to average all metrics across
                all workers into one dict. If a metric is a non-numerical
                value (or nested dictionaries), one value will be randomly
                selected among the workers. If False, returns a list of dicts.
        :param info: (dict) Optional dictionary passed to the training
                operator for ``train_epoch`` and ``train_batch``.

        :return
            (dict | list) A dictionary of metrics for training.
                You can provide custom metrics by passing in a custom
                ``training_operator_cls``. If ``reduce_results=False``,
                this will return a list of metric dictionaries whose
                length will be equal to ``num_workers``.
        """
        if not callable(data_creator):
            raise ValueError(
                "Must provide a callable data_creator, "
                "but got a data_creator of type: {}".format(type(data_creator)))

        success, worker_stats = self._train_epochs(data_creator,
                                                   epochs=epochs,
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

    def _train_epochs(self, data_creator, epochs=1, profile=False, info=None):
        params = dict(data_creator=data_creator, epochs=epochs, profile=profile, info=info)
        remote_worker_stats = []
        for i, w in enumerate(self.remote_workers):
            stats = w.train_epochs.remote(**params)
            remote_worker_stats.append(stats)

        success = check_for_failure(remote_worker_stats)
        if success:
            return success, ray.get(remote_worker_stats)
        else:
            return success, None

    def validate(self, data_creator, num_steps=None, profile=False, info=None):
        """Evaluates the model on the validation data set.

        :param data_creator: (callable) a funtion that takes a config dict as input
                  and return a data loader containing the validation data.
        :param num_steps: (int) Number of batches to compute update steps on.
               This corresponds also to the number of times
                ``TrainingOperator.validate_batch`` is called.
        :param profile: (bool) Returns time stats for the evaluation procedure.
        :param info: (dict) Optional dictionary passed to the training
                operator for `validate` and `validate_batch`.
        :return: A dictionary of metrics for validation.
                You can provide custom metrics by passing in a custom
                ``training_operator_cls``.
        """
        if not callable(data_creator):
            raise ValueError(
                "Must provide a callable data_creator, "
                "but got a data_creator of type: {}".format(type(data_creator)))

        params = dict(data_creator=data_creator,
                      num_steps=num_steps,
                      profile=profile,
                      info=info)

        remote_worker_stats = [
            w.validate.remote(**params) for w in self.remote_workers
        ]
        return self._process_stats(ray.get(remote_worker_stats))

    def get_model(self):
        """Returns the learned model(s)."""
        state = self.get_state_dict()
        model = self.model_creator(self.config)
        model_state = state["models"][0]
        model.load_state_dict(model_state)
        return model.module if hasattr(model, "module") else model

    def save(self, checkpoint):
        """Saves the Estimator state to the provided checkpoint path.

        :param checkpoint: (str) Path to target checkpoint file.
        """
        state_dict = self.get_state_dict()
        torch.save(state_dict, checkpoint)
        return checkpoint

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

    def load(self, checkpoint):
        """Loads the Estimator and all workers from the provided checkpoint.

        :param checkpoint: (str) Path to target checkpoint file.
        """
        state_dict = torch.load(checkpoint)
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

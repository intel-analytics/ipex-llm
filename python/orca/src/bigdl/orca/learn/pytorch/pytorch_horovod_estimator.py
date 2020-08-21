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

from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.learn.pytorch.torch_runner import TorchRunner
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from zoo.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner, HorovodWorker
from zoo.ray import RayContext
import ray

logger = logging.getLogger(__name__)


class TorchWorker(HorovodWorker, TorchRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_components(self):
        import horovod.torch as hvd

        logger.debug("Creating model")
        self.models = self.model_creator(self.config)
        if not isinstance(self.models, collections.Iterable):
            self.models = [self.models]
        else:
            raise ValueError("only support single model for now")

        assert all(isinstance(model, nn.Module) for model in self.models), (
            "All models must be PyTorch models: {}.".format(self.models))

        logger.debug("Creating optimizer.")
        self.optimizers = self.optimizer_creator(self.given_models,
                                                 self.config)
        if not isinstance(self.optimizers, collections.Iterable):
            hvd.broadcast_parameters(self.models[0].state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizers, root_rank=0)
            parameters = self.models[0].named_parameters()
            self.optimizers = hvd.DistributedOptimizer(self.optimizers,
                                                       named_parameters=parameters)
            self.optimizers = [self.optimizers]
        else:
            raise ValueError("only support one optimizer for now")

        self._create_schedulers_if_available()
        self._create_loss()

    def with_sampler(self, loader):
        import horovod.torch as hvd
        # Automatically set the DistributedSampler
        data_loader_args = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "shuffle": False,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "sampler": DistributedSampler(loader.dataset,
                                          num_replicas=hvd.size(),
                                          rank=hvd.rank())
        }
        return DataLoader(**data_loader_args)

    def setup_ddp_and_operator(self):
        """Runs distributed coordination components.

        This helps avoid timeouts due to creator functions (perhaps
        downloading data or models).
        """
        import horovod.torch as hvd

        self.training_operator = self.training_operator_cls(
            self.config,
            models=self.models,
            optimizers=self.optimizers,
            criterion=self.criterion,
            world_rank=hvd.rank(),
            schedulers=self.schedulers,
            use_tqdm=self.use_tqdm)

    def load_state_stream(self, byte_obj):
        """Loads a bytes object the training state dict.

        This is needed because we don't want to deserialize the tensor
        onto the same device (which is from the driver process). We want to
        map it onto the actor's specific device.

        From: github.com/pytorch/pytorch/issues/10622#issuecomment-474733769
        """
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(
            _buffer,
            map_location="cpu")
        return self.load_state_dict(state_dict)

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


class PyTorchHorovodEstimator(HorovodRayRunner):
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
            workers_per_node=1):
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

        self.param = dict(
            model_creator=self.model_creator,
            optimizer_creator=self.optimizer_creator,
            loss_creator=self.loss_creator,
            scheduler_creator=self.scheduler_creator,
            training_operator_cls=self.training_operator_cls,
            scheduler_step_freq=self.scheduler_step_freq,
            use_tqdm=self.use_tqdm,
            config=worker_config)
        super().__init__(RayContext.get(), worker_cls=TorchWorker,
                         worker_param=self.param, workers_per_node=workers_per_node)

        def setup_pytorch():
            import torch
            torch.set_num_threads(self.cores_per_node)
            print("Worker initialized")

        self.run(setup_pytorch)

        def setup_horovod():
            import horovod.torch as hvd
            hvd.init()
            print("Horovod initialized")

        self.run(setup_horovod)

        # Runs the creator functions.
        remote_component_setup = [
            worker.setup_components.remote()
            for i, worker in enumerate(self.remote_workers)
        ]
        # Get setup tasks in order to throw errors on failure
        ray.get(remote_component_setup)

        # Runs code that requires all creator functions to have run.
        remote_operator_setups = [
            worker.setup_ddp_and_operator.remote()
            for worker in self.remote_workers
        ]
        # Get setup tasks in order to throw errors on failure
        ray.get(remote_operator_setups)

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

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

from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import check_for_failure
from ray.util.sgd.torch.distributed_torch_runner import DistributedTorchRunner
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from zoo.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner, HorovodWorker
from zoo.ray import RayContext
import ray

logger = logging.getLogger(__name__)


class TorchWorker(HorovodWorker, DistributedTorchRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, backend="gloo", **kwargs)

    def _setup_distributed_pytorch(self, url, world_rank, world_size):
        import horovod.torch as hvd
        hvd.init()

    def _setup_training(self):
        import horovod.torch as hvd
        logger.debug("Creating model")
        self.models = self.model_creator(self.config)
        if not isinstance(self.models, collections.Iterable):
            self.models = [self.models]
        else:
            raise ValueError("not support single model for now")

        assert all(isinstance(model, nn.Module) for model in self.models), (
            "All models must be PyTorch models: {}.".format(self.models))
        if torch.cuda.is_available():
            self.models = [model.cuda() for model in self.models]

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
        self._try_setup_apex()

        self._create_loss()
        self._initialize_dataloaders()

        self.training_operator = self.training_operator_cls(
            self.config,
            models=self.models,
            optimizers=self.optimizers,
            criterion=self.criterion,
            train_loader=self.train_loader,
            validation_loader=self.validation_loader,
            world_rank=hvd.rank(),
            schedulers=self.schedulers,
            use_fp16=self.use_fp16,
            use_tqdm=self.use_tqdm)

    def _initialize_dataloaders(self):
        import horovod.torch as hvd
        super(DistributedTorchRunner, self)._initialize_dataloaders()

        def with_sampler(loader):
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

        if isinstance(self.train_loader, DataLoader):
            self.train_loader = with_sampler(self.train_loader)

        if self.validation_loader and isinstance(self.validation_loader,
                                                 DataLoader):
            self.validation_loader = with_sampler(self.validation_loader)


class PyTorchHorovodEstimator(HorovodRayRunner):
    def __init__(
            self,
            *,
            model_creator,
            data_creator,
            optimizer_creator,
            loss_creator=None,
            scheduler_creator=None,
            training_operator_cls=TrainingOperator,
            initialization_hook=None,
            config=None,
            scheduler_step_freq="batch"
    ):

        if not (callable(model_creator) and callable(optimizer_creator)
                and callable(data_creator)):
            raise ValueError(
                "Must provide a callable model_creator, optimizer_creator, "
                "and data_creator.")

        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.data_creator = data_creator
        self.scheduler_creator = scheduler_creator
        self.training_operator_cls = training_operator_cls
        self.scheduler_step_freq = scheduler_step_freq

        if not training_operator_cls and not loss_creator:
            raise ValueError("If a loss_creator is not provided, you must "
                             "provide a custom training operator.")

        self.initialization_hook = initialization_hook
        self.config = {} if config is None else config

        self.param = dict(
            model_creator=self.model_creator,
            data_creator=self.data_creator,
            optimizer_creator=self.optimizer_creator,
            loss_creator=self.loss_creator,
            scheduler_creator=self.scheduler_creator,
            training_operator_cls=self.training_operator_cls,
            scheduler_step_freq=self.scheduler_step_freq)
        super().__init__(RayContext.get(), worker_cls=TorchWorker, worker_param=self.param)

        def init_func():
            import torch
            torch.set_num_threads(self.cores_per_node)
            print("Worker initialized")

        self.run(init_func)
        remote_setups = [
            worker.setup.remote(None, None, None)
            for i, worker in enumerate(self.remote_workers)
        ]
        # Get setup tasks in order to throw errors on failure
        ray.get(remote_setups)

    def train(self,
              num_steps=None,
              profile=False,
              reduce_results=True,
              info=None):

        success, worker_stats = self._train_epoch(
            num_steps=num_steps, profile=profile, info=info)

        if reduce_results:
            return self._process_stats(worker_stats)
        else:
            return worker_stats

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

    def _train_epoch(self, num_steps=None, profile=False, info=None):
        params = dict(num_steps=num_steps, profile=profile, info=info)

        remote_worker_stats = [
            w.train_epoch.remote(**params) for w in self.remote_workers
        ]

        success = check_for_failure(remote_worker_stats)
        if success:
            return success, ray.get(remote_worker_stats)

        return success, None

    def validate(self, num_steps=None, profile=False, info=None):
        """Evaluates the model on the validation data set.

        Args:
            num_steps (int): Number of batches to compute update steps on.
                This corresponds also to the number of times
                ``TrainingOperator.validate_batch`` is called.
            profile (bool): Returns time stats for the evaluation procedure.
            info (dict): Optional dictionary passed to the training
                operator for `validate` and `validate_batch`.

        Returns:
            A dictionary of metrics for validation.
                You can provide custom metrics by passing in a custom
                ``training_operator_cls``.
        """
        params = dict(num_steps=num_steps, profile=profile, info=info)

        remote_worker_stats = [
            w.validate.remote(**params) for w in self.remote_workers
        ]
        return self._process_stats(ray.get(remote_worker_stats))

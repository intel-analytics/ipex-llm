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
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/torch/torch_runner.py

from filelock import FileLock
import logging
import inspect
import io
import itertools
import os
import tempfile
import torch
import torch.nn as nn

import ray
from zoo.orca.learn.pytorch.constants import SCHEDULER_STEP, NUM_STEPS
from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.learn.pytorch import utils

logger = logging.getLogger(__name__)

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


class TorchRunner:
    """Manages a PyTorch model for training."""

    def __init__(self,
                 model_creator,
                 data_creator,
                 optimizer_creator,
                 loss_creator=None,
                 scheduler_creator=None,
                 training_operator_cls=None,
                 config=None,
                 serialize_data_creation=True,
                 use_tqdm=False,
                 scheduler_step_freq=None):
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.data_creator = data_creator
        self.scheduler_creator = scheduler_creator
        self.training_operator_cls = training_operator_cls or TrainingOperator
        self.config = {} if config is None else config

        self.timers = utils.TimerCollection()
        self.epochs = 0
        self.models = None
        self.optimizers = None
        self.criterion = None
        self.schedulers = None
        self.train_loader = None
        self.validation_loader = None
        self.training_operator = None
        self.serialize_data_creation = serialize_data_creation
        self.use_tqdm = use_tqdm
        self.scheduler_step_freq = scheduler_step_freq

    def _validate_loaders(self, loaders):
        assert loaders, "Loaders need to be returned in data_creator."
        if isinstance(loaders, (tuple, list)):
            if len(loaders) == 1:
                return loaders, None
            elif len(loaders) == 2:
                return loaders
            else:
                raise ValueError(
                    "Number of loaders must be <= 2. Got {}".format(loaders))
        # No great way of checking type otherwise
        return loaders, None

    def _initialize_dataloaders(self):
        logger.debug("Instantiating dataloaders.")
        loaders = None
        if self.serialize_data_creation:
            logger.debug("Serializing the dataloading process.")
            with FileLock(
                    os.path.join(tempfile.gettempdir(), ".raydata.lock")):
                loaders = self.data_creator(self.config)
        else:
            loaders = self.data_creator(self.config)
        train_loader, val_loader = self._validate_loaders(loaders)

        self.train_loader, self.validation_loader = train_loader, val_loader

    def _create_loss(self):
        if not self.loss_creator:
            return
        logger.debug("Creating loss.")
        if inspect.isclass(self.loss_creator) and issubclass(
                self.loss_creator, torch.nn.modules.loss._Loss):
            self.criterion = self.loss_creator()
        else:
            self.criterion = self.loss_creator(self.config)

    def _create_schedulers_if_available(self):
        # Learning rate schedules are optional.
        if not self.scheduler_creator:
            return
        self.schedulers = self.scheduler_creator(self.given_optimizers,
                                                 self.config)

        if not isinstance(self.schedulers, Iterable):
            self.schedulers = [self.schedulers]

    def setup_components(self):
        """Runs the creator functions without any distributed coordination."""
        logger.debug("Loading data.")
        if self.data_creator and callable(self.data_creator):
            self._initialize_dataloaders()

        logger.debug("Creating model")
        self.models = self.model_creator(self.config)
        if not isinstance(self.models, Iterable):
            self.models = [self.models]
        assert all(isinstance(model, nn.Module) for model in self.models), (
            "All models must be PyTorch models: {}.".format(self.models))

        logger.debug("Creating optimizer.")
        self.optimizers = self.optimizer_creator(self.given_models,
                                                 self.config)
        if not isinstance(self.optimizers, Iterable):
            self.optimizers = [self.optimizers]

        self._create_schedulers_if_available()
        self._create_loss()

    def setup_operator(self):
        """Create the training operator."""
        self.training_operator = self.training_operator_cls(
            self.config,
            models=self.models,
            optimizers=self.optimizers,
            criterion=self.criterion,
            train_loader=self.train_loader,
            validation_loader=self.validation_loader,
            world_rank=0,
            schedulers=self.schedulers,
            use_tqdm=self.use_tqdm)

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return utils.find_free_port()

    def train_epoch(self,
                    num_steps=None,
                    profile=False,
                    info=None,
                    iterator=None):
        """Runs a training epoch and updates the model parameters."""
        logger.debug("Begin Training Step {}".format(self.epochs + 1))
        info = info or {}
        self._toggle_profiling(profile=profile)

        info.update({
            NUM_STEPS: num_steps,
            SCHEDULER_STEP: self.scheduler_step_freq
        })
        with self.timers.record("train_epoch"):
            if iterator is None:
                iterator = iter(self.train_loader)
            else:
                # Dataset will provide us with a list of tuples but we
                # need two lists.
                def format_batch(batch):
                    features, targets = zip(*batch)
                    return torch.cat(features), torch.cat(targets)

                iterator = map(format_batch, iterator)
            if num_steps:
                iterator = itertools.islice(iterator, num_steps)
            train_stats = self.training_operator.train_epoch(iterator, info)

        self.epochs += 1
        # This is so that `epochs` is first in ordering.
        stats = dict(epoch=self.epochs, **train_stats)
        if profile:
            stats.update(profile=self.timers.stats())
        return stats

    def validate(self, num_steps=None, profile=False, info=None):
        """Evaluates the model on the validation data set."""
        if self.validation_loader is None:
            raise ValueError("No validation dataloader provided.")
        info = info or {}
        self._toggle_profiling(profile=profile)

        with self.timers.record("validation"):
            iterator = self.validation_loader
            if num_steps:
                iterator = itertools.islice(
                    iter(self.validation_loader), num_steps)
            validation_stats = self.training_operator.validate(
                iterator, info=info)
        if profile:
            validation_stats.update(profile=self.timers.stats())
        return validation_stats

    def _toggle_profiling(self, profile=False):
        """Enables/Disables and resets timing profiles."""
        if profile:
            self.timers.enable()
            self.timers.reset()
        else:
            self.timers.disable()
        self.training_operator._set_timers(self.timers)

    def state_dict(self):
        """Returns the state of the runner."""
        state = {
            "epoch": self.epochs,
            "operator": self.training_operator.state_dict(),
            "models": [model.state_dict() for model in self.models],
            "optimizers": [opt.state_dict() for opt in self.optimizers]
        }
        if self.schedulers:
            state.update({
                "schedulers": [
                    scheduler.state_dict() for scheduler in self.schedulers
                ]
            })
        return state

    def load_state_dict(self, state):
        """Sets the state of the model."""
        for model, state_dict in zip(self.models, state["models"]):
            model.load_state_dict(state_dict)
        for optimizer, state_dict in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(state_dict)
        if self.schedulers:
            for scheduler, state_dict in zip(self.schedulers,
                                             state["schedulers"]):
                scheduler.load_state_dict(state_dict)

        self.epochs = state["epoch"]
        self.training_operator.load_state_dict(state_dict)

    def state_stream(self):
        """Returns a bytes object for the state dict."""
        state_dict = self.state_dict()
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        return _buffer.getvalue()

    def load_state_stream(self, byte_obj):
        """Loads a bytes object the training state dict."""
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(_buffer)
        return self.load_state_dict(state_dict)

    def apply(self, fn):
        return fn()

    def apply_operator(self, fn):
        return fn(self.training_operator)

    def shutdown(self):
        """Attempts to shut down the worker."""
        del self.training_operator
        del self.validation_loader
        del self.train_loader
        del self.criterion
        del self.optimizers
        del self.models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def given_models(self):
        if len(self.models) > 1:
            return self.models
        else:
            return self.models[0]

    @property
    def given_optimizers(self):
        if len(self.optimizers) > 1:
            return self.optimizers
        else:
            return self.optimizers[0]

    @property
    def given_schedulers(self):
        if not self.schedulers:
            return self.schedulers
        if len(self.schedulers) > 1:
            return self.schedulers
        else:
            return self.schedulers[0]

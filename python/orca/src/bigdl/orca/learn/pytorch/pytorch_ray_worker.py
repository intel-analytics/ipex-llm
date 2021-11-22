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
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/torch/torch_runner.py

import ray
from bigdl.orca.learn.pytorch.utils import find_free_port
from bigdl.orca.learn.pytorch.torch_runner import TorchRunner
import torch.nn as nn


import logging
logger = logging.getLogger(__name__)

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


class PytorchRayWorker(TorchRunner):
    """Manages a PyTorch model for training."""

    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 loss_creator=None,
                 metrics=None,
                 scheduler_creator=None,
                 training_operator_cls=None,
                 config=None,
                 use_tqdm=False,
                 scheduler_step_freq=None):
        super().__init__(model_creator, optimizer_creator, loss_creator, metrics, scheduler_creator,
                         training_operator_cls, config, use_tqdm, scheduler_step_freq)

        self.backend = "torch-local"
        self.rank = 0
        self.size = 0

    def setup_horovod(self):
        import horovod.torch as hvd
        hvd.init()
        self.backend = "horovod"
        self.rank = hvd.rank()
        self.size = hvd.size()
        self.setup_components_horovod()
        self.setup_operator(self.models)

    def setup_address(self):
        ip = ray._private.services.get_node_ip_address()
        port = find_free_port()
        return f"tcp://{ip}:{port}"

    def setup_components_horovod(self):
        import horovod.torch as hvd

        logger.debug("Creating model")
        self.models = self.model_creator(self.config)
        if not isinstance(self.models, Iterable):
            self.models = [self.models]
        else:
            raise ValueError("only support single model for now")

        assert all(isinstance(model, nn.Module) for model in self.models), (
            "All models must be PyTorch models: {}.".format(self.models))

        logger.debug("Creating optimizer.")
        self.optimizers = self.optimizer_creator(self.given_models,
                                                 self.config)
        if not isinstance(self.optimizers, Iterable):
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

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray._private.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return utils.find_free_port()

    def predict(self, data_creator, batch_size=32, profile=False):
        """Evaluates the model on the validation data set."""
        config = self.config.copy()
        self._toggle_profiling(profile=profile)

        shards_ref = data_creator(config, batch_size)
        if not isinstance(shards_ref, ray.ObjectID):
            raise ValueError("Only xshards is supported for predict")

        partition = ray.get(shards_ref)
        return super().predict(partition=partition, batch_size=batch_size, profile=profile)

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
        self.training_operator.load_state_dict(state["operator"])

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

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
from collections import OrderedDict
import inspect
from typing import List
import torch

from torchmetrics.metric import Metric
from pytorch_lightning import LightningModule
from torch import nn, Tensor, fx
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from bigdl.nano.utils.log4Error import invalidInputError


class LightningModuleFromTorch(LightningModule):
    """
    A wrapper LightningMoudle for common PyTorch models.

    This class implements common methods in LightningModule, so that classic pytorch
    supervised learning model only needs to supply module, loss and optimizer to create
    a LightningModule.
    """

    def __init__(self, model: nn.Module, loss: _Loss = None, optimizer: Optimizer = None,
                 scheduler: _LRScheduler = None,
                 metrics: List[Metric] = None):
        """
        Create a LightningMoudle that integrates pytorch modules, loss, optimizer.

        :param model:       Pytorch model to be converted.
        :param loss:        A torch loss function.
        :param optimizer:   A torch optimizer.
        :param scheduler:   A torch scheduler.
        :param metrics:     A list of metrics to calculate accuracy of the model.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

    def forward(self, *args):
        """Same as torch.nn.Module.forward()."""
        nargs = len(inspect.getfullargspec(self.model.forward).args[1:])
        if isinstance(args, fx.Proxy):
            args = [args[i] for i in range(nargs)]
        else:
            args = args[:nargs]
        return self.model(*args)

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""
        invalidInputError(self.loss, "Loss must not be None for training.")
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        """Define a single training step, return a loss tensor."""
        y_hat = self(*batch)
        loss = self.loss(y_hat, batch[-1])  # use last output as target
        self.log("train/loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Define a single validation step."""
        y_hat = self(*batch)
        if self.loss:
            loss = self.loss(y_hat, batch[-1])  # use last output as target
            self.log("val/loss", loss, on_epoch=True,
                     prog_bar=True, logger=True)
        if self.metrics:
            acc = {"val/{}_{}".format(type(metric).__name__, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Define a single test step."""
        y_hat = self(*batch)
        if self.metrics:
            acc = {"test/{}_{}".format(type(metric).__name__, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        """Define a single predict step."""
        return self(*batch)

    def configure_optimizers(self):
        """Setup the optimizers for this module, and return optimizers and schedulers."""
        optimizers = [self.optimizer]
        schedulers = []
        if self.scheduler:
            schedulers.append(self.scheduler)
        return optimizers, schedulers

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Same as LightningModule.load_state_dict, execept falling back to pytorch."""
        try:
            super().load_state_dict(state_dict)
        except RuntimeError:
            self.model.load_state_dict(state_dict)

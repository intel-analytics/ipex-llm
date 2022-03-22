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
from typing import List

from torchmetrics.metric import Metric
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LightningModuleFromTorch(LightningModule):
    def __init__(self, model: nn.Module, loss: _Loss = None, optimizer: Optimizer = None,
                 scheduler: _LRScheduler = None,
                 metrics: List[Metric] = None):
        """
        Integrate pytorch modules, loss, optimizer to pytorch-lightning model.

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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _forward(self, batch):
        # Handle different numbers of input for various models
        nargs = self.model.forward.__code__.co_argcount
        return self(*(batch[:nargs - 1]))

    def on_train_start(self) -> None:
        assert self.loss, "Loss must not be None for training."
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        y_hat = self._forward(batch)
        loss = self.loss(y_hat, batch[-1])  # use last output as target
        self.log("train/loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self._forward(batch)
        if self.loss:
            loss = self.loss(y_hat, batch[-1])  # use last output as target
            self.log("val/loss", loss, on_epoch=True,
                     prog_bar=True, logger=True)
        if self.metrics:
            acc = {"val/{}_{}".format(type(metric).__name__, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        y_hat = self._forward(batch)
        if self.metrics:
            acc = {"test/{}_{}".format(type(metric).__name__, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        return self(*batch)

    def configure_optimizers(self):
        optimizers = [self.optimizer]
        schedulers = []
        if self.scheduler:
            schedulers.append(self.scheduler)
        return optimizers, schedulers

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        try:
            super().load_state_dict(state_dict)
        except RuntimeError:
            self.model.load_state_dict(state_dict)

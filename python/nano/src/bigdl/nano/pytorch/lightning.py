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


class LightningModuleFromTorch(LightningModule):
    def __init__(self, model: nn.Module, loss: _Loss = None, optimizer: Optimizer = None,
                 metrics: List[Metric] = None):
        """
        Integrate pytorch modules, loss, optimizer to pytorch-lightning model.

        :param model:       Pytorch model to be converted.
        :param loss:        A torch loss function.
        :param optimizer:   A torch optimizer.
        :param metrics:     A list of metrics to calculate accuracy of the model.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _forward(self, batch):
        # Handle different numbers of input for various models
        nargs = self.model.forward.__code__.co_argcount
        return self.model(*(batch[:nargs - 1]))

    def training_step(self, batch, batch_idx):
        y_hat = self._forward(batch)
        loss = self.loss(y_hat, batch[-1])  # use last output as target
        self.log("train/loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self._forward(batch)
        loss = self.loss(y_hat, batch[-1])  # use last output as target
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics:
            acc = {"val/" + type(metric).__name__: metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)
        else:
            acc = None
        return loss, acc

    def test_step(self, batch, batch_idx):
        y_hat = self._forward(batch)
        loss = self.loss(y_hat, batch[-1])  # use last output as target
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics:
            acc = {"test/" + type(metric).__name__: metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)
        else:
            acc = None
        return loss, acc

    def configure_optimizers(self):
        return self.optimizer

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        try:
            super().load_state_dict(state_dict)
        except RuntimeError:
            self.model.load_state_dict(state_dict)

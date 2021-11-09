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
from pytorch_lightning import LightningModule
from torch import nn
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


class LightningModuleFromTorch(LightningModule):
    def __init__(self, model: nn.Module, loss: _Loss, optimizer: Optimizer):
        """
        Integrate pytorch modules, loss, optimizer to pytorch-lightning model.

        :param model:       Pytorch model to be converted.
        :param loss:        A torch loss function.
        :param optimizer:   A torch optimizer.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def _forward(self, batch):
        # Handle different numbers of input for various models
        nargs = self.model.forward.__code__.co_argcount
        return self.model(*(batch[:nargs-1]))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._forward(batch)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._forward(batch)
        loss = self.loss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._forward(batch)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return self.optimizer

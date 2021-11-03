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


class LightningModuleFromTorch(LightningModule):
    def __init__(self, model: nn.Module, loss: _Loss, optimizer: torch.optim):
        """
        Integrate pytorch modules, loss, optimizer to pytorch-lightning model.

        Args:
            model: pytorch model to be converted.
            loss: torch loss function.
            optimizer: torch optimizer.

        Returns: LightningModule Object
        """
        super().__init__()
        self.copy(model)
        self._loss = loss
        self._optimizer = optimizer

    def copy(self, torch_model):
        for name, child in torch_model._modules.items():
            setattr(self, name, child)
        self.forward = torch_model.forward

    @property
    def loss(self):
        return self._loss

    def _forward(self, batch):
        # Handle different numbers of input for various models
        nargs = self.forward.__code__.co_argcount
        return self.forward(*(batch[:nargs - 1]))

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
        return self._optimizer



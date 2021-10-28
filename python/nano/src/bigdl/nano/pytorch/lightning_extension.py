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
import pytorch_lightning as pl
import torch
from torch.nn.modules.loss import _Loss


def to_lightning(loss: _Loss, optimizer: torch.optim, **opt_args):
    """
    A decorator on torch module creator, returns a pytorch-lightning model.

    Args:
        loss: torch loss function.
        optimizer: torch optimizer.
        **opt_args: arguments for optimizer.

    Returns: Decorator function on class or function

    """

    def from_torch(creator):
        class LightningModel(pl.LightningModule):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.model = creator(*args, **kwargs)
                self.loss = loss
                self.optimizer = optimizer(self.model.parameters(), **opt_args)

            def forward(self, batch):
                # Handle different numbers of input for various models
                nargs = self.model.forward.__code__.co_argcount
                return self.model(*batch[:nargs - 1])

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(batch)
                loss = self.loss(y_hat, y)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(batch)
                loss = self.loss(y_hat, y)
                return loss

            def test_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(batch)
                loss = self.loss(y_hat, y)
                return loss

            def configure_optimizers(self):
                return self.optimizer

        return LightningModel

    return from_torch

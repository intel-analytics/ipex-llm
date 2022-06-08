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

from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.utils.log4Error import invalidInputError
from torch import nn
from pytorch_lightning import LightningModule
import bigdl.nano.automl.hpo as hpo
import warnings
import torch
from torch.utils.data import TensorDataset, DataLoader


@hpo.plmodel()
class GenericLightningModule(LightningModuleFromTorch):
    """A generic LightningMoudle for light-weight HPO."""

    def __init__(self,
                 model_creator, optim_creator, loss_creator, data,
                 model_config={}, data_config={}, optim_config={}, loss_config={},
                 batch_size=32,
                 epochs=1,
                 validation_data=None,
                 scheduler=None, metrics=None,
                 num_processes=1):
        """
        Create LightningModule that exposes hyper parameters in init arguments.

        :param model_creator:   the model creator function.
        :param model_config:    model related configurations - argument of model_creator.
        :param data_config:     data related configurations - argument of model_creator.
        :param optim_creator:   the optimizer creator function.
        :param optim_config:    optim configurations - argument of optim_creator.
        :param loss_creator:    the loss creator function.
        :param loss_config:     the loss configurations - argument of loss_creator.
        :param data:            the train data
        :param validatoni
        :param scheduler:       learning rate scheduler.
        :param metrics:         list of metrics to calculate accuracy of the model.
        """
        super().__init__()
        self.model_creator = model_creator
        self.optim_creator = optim_creator
        self.loss_creator = loss_creator
        self.data = data
        self.validation_data = validation_data

        # search spaces can be specified in configs
        self.model_config = model_config
        self.data_config = data_config
        self.optim_config = optim_config
        self.loss_config = loss_config

        # set batch size
        if batch_size % num_processes != 0:
            warnings.warn("'batch_size' cannot be divided with no remainder by "
                          "'num_processes'. Change 'batch_size' to {} for "
                          "'num_processes' = {}".
                          format(batch_size, num_processes))
        self.batch_size = batch_size // num_processes

        self.epochs = epochs
        self.scheduler = scheduler
        self.metrics = metrics

        self.model = self.model_creator({**self.model_config, **self.data_config})
        self.loss = self.loss_creator(self.loss_config)
        self.optimizer = self.optimizer_creator(self.model, self.optim_config)

        invalidInputError(isinstance(self.model, nn.Module) and not
                          isinstance(self.model, LightningModule),
                          "The created model must be instance of nn.Module but got {}"
                          .format(self.model.__class__))

        super().__init__(model=self.model, loss=self.loss, optimizer=self.optimizer,
                         scheduler=scheduler, metrics=metrics)

    def train_dataloader(self):
        """Create the train data loader."""
        return DataLoader(TensorDataset(torch.from_numpy(self.data[0]),
                                        torch.from_numpy(self.data[1])),
                          batch_size=max(1, self.batch_size),
                          shuffle=True)

    def val_dataloader(self):
        """Create the validation data loader."""
        return DataLoader(TensorDataset(torch.from_numpy(self.validation_data[0]),
                                        torch.from_numpy(self.validation_data[1])),
                          batch_size=max(1, self.batch_size),
                          shuffle=True)

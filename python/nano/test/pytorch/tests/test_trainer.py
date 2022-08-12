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


import copy
import os
import shutil
from unittest import TestCase

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import train_with_linear_top_layer
from torch import nn

from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


class LitResNet18(LightningModule):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.classify = nn.Sequential(backbone, head)

    def forward(self, *args):
        return self.classify(args[0])


class TestTrainer(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
    user_defined_pl_model = LitResNet18(10)

    def test_resnet18(self):
        resnet18 = vision.resnet18(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet18, batch_size, num_workers, data_dir)

    def test_trainer_compile(self):
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        trainer.fit(pl_model, self.train_loader)

    def test_trainer_precision_bf16(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = Trainer.compile(model, loss, optimizer)
        if TORCH_VERSION_LESS_1_10:
            trainer = Trainer(max_epochs=1, precision=64)
            trainer.fit(pl_model, self.train_loader)
            assert isinstance(trainer.strategy.precision_plugin, DoublePrecisionPlugin)
            assert optimizer.param_groups[0]['params'][0].dtype is torch.float64
        else:
            trainer = Trainer(max_epochs=1, precision='bf16')
            trainer.fit(pl_model, self.train_loader)
            assert isinstance(trainer.strategy.precision_plugin, NativeMixedPrecisionPlugin)
            # model is not converted to bfloat16 precision
            input = TensorDataset(torch.rand(1, 3, 32, 32))
            train_loader = DataLoader(input)
            y_hat = trainer.predict(pl_model, train_loader)
            assert y_hat[0].dtype is torch.bfloat16

    def test_trainer_save_load(self):
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        trainer.save(pl_model, "saved_model")
        assert len(os.listdir('saved_model')) > 0

        # save original parameters
        original_state_dict = copy.deepcopy(pl_model.state_dict())

        # update paramters
        trainer.fit(pl_model, self.train_loader)

        loaded_pl_model = trainer.load("saved_model", pl_model)
        loaded_state_dict = loaded_pl_model.state_dict()
        # check if parameters are updated
        for k in original_state_dict.keys():
            assert (original_state_dict[k] == loaded_state_dict[k]).all()
        shutil.rmtree('saved_model')


if __name__ == '__main__':
    pytest.main([__file__])

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

import os
from unittest import TestCase

import torch
import torchmetrics
from torch import nn

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision

num_classes = 10
batch_size = 256
num_workers = 0
data_dir = "/tmp/data"


class ResNet18(nn.Module):
    def __init__(self, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = torch.nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


model = ResNet18(pretrained=False, include_top=False, freeze=True)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


class TestLightningModule(TestCase):

    def test_resnet18(self):
        pl_model = LightningModule(
            model, loss, optimizer,
            metrics=[torchmetrics.F1Score('multiclass', num_classes=num_classes),
                     torchmetrics.Accuracy('multiclass', num_classes=num_classes)]
        )
        data_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
        trainer = Trainer(max_epochs=4, log_every_n_steps=1)
        trainer.fit(pl_model, data_loader, data_loader)
        trainer.validate(pl_model, data_loader)
        trainer.test(pl_model, data_loader)
        trainer.predict(pl_model, data_loader)

    def test_load_state_dict_from_torch(self):
        torch.save(model.state_dict(), "resnet18_test.pth")
        pl_model = LightningModule(model, loss, optimizer)
        state_dict = torch.load("resnet18_test.pth")
        pl_model.load_state_dict(state_dict)

    def test_load_state_dict_from_lightning(self):
        pl_model = LightningModule(model, loss, optimizer)
        torch.save(pl_model.state_dict(), "lightning_resnet18_test.pth")
        state_dict = torch.load("lightning_resnet18_test.pth")
        pl_model.load_state_dict(state_dict)

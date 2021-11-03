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
from test._train_torch_lightning import train_torch_lightning
from torch import nn

from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.vision.models import vision

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")

loss = nn.CrossEntropyLoss()


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = torch.nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


class TestLightningModuleFromTorch(TestCase):
    num_classes = 10

    def test_resnet18_ipex(self):
        model = ResNet18(10, pretrained=True, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = LightningModuleFromTorch(model, loss, optimizer)
        train_torch_lightning(pl_model, batch_size, num_workers, data_dir,
                              use_orca_lite_trainer=True)

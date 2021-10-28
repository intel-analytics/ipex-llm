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

from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.vision.models.lightning_extension import to_lightning

config = {
    'lr': 0.01,
    'optim': 'Adam',
}
batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")


def loss_creator(config):
    return nn.CrossEntropyLoss()


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(), lr=config.get("lr", 0.001))


@to_lightning(loss_creator, optimizer_creator, config)
def resnet18(num_classes, pretrained=True, include_top=False, freeze=True):
    backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
    output_size = backbone.get_output_size()
    head = nn.Linear(output_size, num_classes)
    return torch.nn.Sequential(backbone, head)


class TestModelsLightningSupport(TestCase):
    num_classes = 10

    def test_resnet18_ipex(self):
        pl_resnet18 = resnet18(10, pretrained=True, include_top=False, freeze=True)
        train_torch_lightning(pl_resnet18, batch_size, num_workers, data_dir,
                              use_orca_lite_trainer=True)

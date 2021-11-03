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


import pytest
import os
from unittest import TestCase

import torch
from torch import nn

from _train_torch_lightning import train_torch_lightning
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.vision.models import vision
from test._train_torch_lightning import train_with_linear_top_layer

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestModelsVision(TestCase):

    def test_resnet18_ipex(self):
        resnet18 = vision.resnet18(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet18, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_resnet34_ipex(self):
        resnet34 = vision.resnet34(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet34, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_resnet50_ipex(self):
        resnet50 = vision.resnet50(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet50, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_mobilenet_v3_large_ipex(self):
        mobilenet = vision.mobilenet_v3_large(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_mobilenet_v3_small_ipex(self):
        mobilenet = vision.mobilenet_v3_small(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_mobilenet_v2_ipex(self):
        mobilenet = vision.mobilenet_v2(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_shufflenet_ipex(self):
        shufflenet = vision.shufflenet_v2_x1_0(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            shufflenet, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_trainer_compile(self):
        class ResNet18(nn.Module):
            def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
                super().__init__()
                backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
                output_size = backbone.get_output_size()
                head = nn.Linear(output_size, num_classes)
                self.model = nn.Sequential(backbone, head)

            def forward(self, x):
                return self.model(x)

        model = ResNet18(10, pretrained=True, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = Trainer.compile(model, loss, optimizer)
        train_torch_lightning(pl_model, batch_size, num_workers, data_dir,
                              use_orca_lite_trainer=True)


if __name__ == '__main__':
    pytest.main([__file__])

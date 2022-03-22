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

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule

import torchmetrics
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.vision.models import vision

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")

_data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])


def _create_data_loader(dir, batch_size, num_workers, transform, subset=50):
    train_set = CIFAR10(root=dir, train=True,
                        download=True, transform=transform)
    # `subset` is the number of subsets. The larger the number, the smaller the training set.
    mask = list(range(0, len(train_set), subset))
    train_subset = torch.utils.data.Subset(train_set, mask)
    data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)
    return data_loader


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


class TestPlugin(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = _create_data_loader(data_dir, batch_size, num_workers, _data_transform)

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(
            os.path.join(os.path.join(os.path.join(test_dir, ".."), ".."), "..")
        )
        os.environ['WORKER'] = project_test_dir
        print(project_test_dir)

    def test_trainer_subprocess_plugin(self):
        pl_model = LightningModuleFromTorch(self.model, self.loss, self.optimizer)
        trainer = Trainer(num_processes=2, distributed_backend="subprocess", max_epochs=4)
        trainer.fit(pl_model, self.train_loader)
        trainer.test(pl_model, self.train_loader)


if __name__ == '__main__':
    pytest.main([__file__])

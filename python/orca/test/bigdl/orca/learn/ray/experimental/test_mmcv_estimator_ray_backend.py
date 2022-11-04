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

import unittest
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch.experimential.mmcv.mmcv_ray_estimator import MMCVRayEstimator


resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../resources")


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_step(self, data, optimizer):
        images, labels = data
        predicts = self(images)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        return {'loss': loss}


def runner_creator(config):
    model = Model()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    logger = get_logger('mmcv')
    # runner is a scheduler to manage the training
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir='./work_dir',
        logger=logger,
        max_epochs=4)

    # learning rate scheduler config
    lr_config = dict(policy='step', step=[2, 3])
    # configuration of optimizer
    optimizer_config = dict(grad_clip=None)
    # save log periodically and multiple hooks can be used simultaneously
    log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        log_config=log_config)

    return runner


def train_dataloader_creator(config):
    # dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_path = os.path.join(resource_path, "cifar10")
    train_set = CIFAR10(
        root=data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=2)
    return train_loader


class TestMMCVRayEstimator(unittest.TestCase):

    estimator = None

    @classmethod
    def setUpClass(cls) -> None:
        init_orca_context(cores=8, memory="8g")
        cls.estimator = MMCVRayEstimator(
            mmcv_runner_creator=runner_creator,
            config={}
        )

    @classmethod
    def tearDownClass(cls) -> None:
        data_path = os.path.join(resource_path, "cifar10")
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

        stop_orca_context()

    def test_fit(self):
        self.estimator.fit([train_dataloader_creator], [('train', 1)])


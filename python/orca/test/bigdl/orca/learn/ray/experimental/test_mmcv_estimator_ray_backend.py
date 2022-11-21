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
import unittest
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger
from bigdl.orca.learn.pytorch.experimential.mmcv.mmcv_ray_estimator import MMCVRayEstimator

resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../resources")

MAX_EPOCH = 4
NUM_SAMPLES = 1000


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

    def train_step(self, data, optimizer, **kwargs):
        features, labels = data
        predicts = self(features)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        return {'loss': loss}


class Model2(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, input_, labels):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        loss = self.loss_fn(y, labels)
        return loss


def batch_processor(model, data, train_mode, **kwargs):
    features, labels = data
    loss = model(features, labels)
    log_vars = dict()
    log_vars["var1"] = 1.0
    return {'loss': loss, 'log_vars': log_vars, "num_samples": features.size(0)}


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
        max_epochs=MAX_EPOCH)

    # learning rate scheduler config
    lr_config = dict(policy='step', step=[2, 3])
    # configuration of optimizer
    optimizer_config = dict(grad_clip=None)
    # save log periodically and multiple hooks can be used simultaneously
    log_config = dict(interval=4, hooks=[dict(type='TextLoggerHook')])
    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        log_config=log_config)

    return runner


def runner_creator_with_batch_processor(config):
    model = Model2()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    logger = get_logger('mmcv')
    # runner is a scheduler to manage the training
    runner = EpochBasedRunner(
        model,
        batch_processor=batch_processor,
        optimizer=optimizer,
        work_dir='./work_dir',
        logger=logger,
        max_epochs=MAX_EPOCH)

    # learning rate scheduler config
    lr_config = dict(policy='step', step=[2, 3])
    # configuration of optimizer
    optimizer_config = dict(grad_clip=None)
    # save log periodically and multiple hooks can be used simultaneously
    log_config = dict(interval=4, hooks=[dict(type='TextLoggerHook')])
    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        log_config=log_config)

    return runner


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, size=1000):
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.x = torch.cat([X1, X2], dim=0)
        Y1 = torch.zeros(size // 2, 1)
        Y2 = torch.ones(size // 2, 1)
        self.y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


def train_dataloader_creator(config):
    train_set = LinearDataset(size=NUM_SAMPLES)
    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    return train_loader


def get_estimator(creator):
    estimator = MMCVRayEstimator(
        mmcv_runner_creator=creator,
        config={}
    )
    return estimator


class TestMMCVRayEstimator(unittest.TestCase):

    def test_run_with_train_step(self):
        estimator = get_estimator(runner_creator)
        epoch_stats = estimator.run([train_dataloader_creator], [('train', 1)])
        self.assertEqual(len(epoch_stats), MAX_EPOCH)

        start_stats = epoch_stats[0]
        end_stats = epoch_stats[-1]
        self.assertEqual(start_stats["num_samples"], NUM_SAMPLES)
        self.assertEqual(end_stats["num_samples"], NUM_SAMPLES)

        dloss = end_stats["loss"] - start_stats["loss"]
        print(f"dLoss: {dloss}")
        assert dloss < 0

    def test_run_with_batch_processor(self):
        estimator = get_estimator(runner_creator_with_batch_processor)
        epoch_stats = estimator.run([train_dataloader_creator], [('train', 1)])
        self.assertEqual(len(epoch_stats), MAX_EPOCH)

        start_stats = epoch_stats[0]
        end_stats = epoch_stats[-1]
        self.assertEqual(start_stats["num_samples"], NUM_SAMPLES)
        self.assertEqual(end_stats["num_samples"], NUM_SAMPLES)
        self.assertEqual(start_stats["var1"], 1.0)
        self.assertEqual(end_stats["var1"], 1.0)

        dloss = end_stats["loss"] - start_stats["loss"]
        print(f"dLoss: {dloss}")
        assert dloss < 0


if __name__ == "__main__":
    pytest.main([__file__])

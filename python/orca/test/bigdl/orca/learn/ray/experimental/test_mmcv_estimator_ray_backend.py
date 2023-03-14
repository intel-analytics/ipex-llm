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
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mmcv.runner import EpochBasedRunner
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.utils import get_logger
from bigdl.orca.learn.pytorch import Estimator

resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../resources")

MAX_EPOCH = 4
NUM_SAMPLES = 1000
LAST_EVAL_LOSS = 0.0
TEMP_WORK_DIR = os.path.join(tempfile.gettempdir(), "mmcv_test_work_dir")


class DistEvalHook(BaseDistEvalHook):

    def _do_evaluate(self, runner):
        """
        when use DistEvalHook with multi worker, make sure the val data is
        correctly split into multi worker.
        """
        worker_nums = 2
        samples_per_worker = NUM_SAMPLES / worker_nums
        actual_samples_per_worker = 0
        for data, label in self.dataloader:
            actual_samples_per_worker += len(data)
        assert samples_per_worker == actual_samples_per_worker


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


def runner_creator(cfg):
    model = cfg['model']
    optimizer = cfg['optimizer']
    batch_processor_fn = cfg['batch_processor']
    logger = get_logger('mmcv')
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        batch_processor=batch_processor_fn,
        work_dir=TEMP_WORK_DIR,
        logger=logger,
        max_epochs=MAX_EPOCH)

    # learning rate scheduler config
    lr_config = cfg['lr_config']
    # configuration of saving checkpoints periodically
    checkpoint_config = cfg['checkpoint_config']
    # configuration of optimizer
    optimizer_config = cfg['optimizer_config']
    # save log periodically and multiple hooks can be used simultaneously
    log_config = cfg['log_config']
    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        log_config=log_config)

    if cfg.get('add_eval_hook'):
        val_set = LinearDataset(size=NUM_SAMPLES)
        val_loader = DataLoader(
            val_set, batch_size=64, shuffle=True, num_workers=2)
        eval_hook = DistEvalHook(val_loader)
        runner.register_hook(eval_hook, priority='LOW')

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


def get_estimator(creator, cfg=None, workers_per_node=1):
    if cfg is None:
        cfg = {}
    estimator = Estimator.from_mmcv(
        mmcv_runner_creator=creator,
        config=cfg,
        workers_per_node=workers_per_node
    )
    return estimator


class TestMMCVRayEstimator(unittest.TestCase):

    def test_run_with_train_step(self):
        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        cfg = dict(
            model=model,
            optimizer=optimizer,
            batch_processor=None,
            lr_config=dict(policy='step', step=[2, 3]),
            optimizer_config=dict(grad_clip=None),
            checkpoint_config=None,
            log_config=dict(interval=4, hooks=[dict(type='TextLoggerHook')])
        )
        estimator = get_estimator(runner_creator, cfg)
        epoch_stats = estimator.run([train_dataloader_creator], [('train', 1)])
        self.assertEqual(len(epoch_stats), MAX_EPOCH)

        start_stats = epoch_stats[0]
        end_stats = epoch_stats[-1]
        self.assertEqual(start_stats["num_samples"], NUM_SAMPLES)
        self.assertEqual(end_stats["num_samples"], NUM_SAMPLES)

        dloss = end_stats["loss"] - start_stats["loss"]
        print(f"dLoss: {dloss}")
        assert dloss < 0

        if os.path.exists(TEMP_WORK_DIR):
            shutil.rmtree(TEMP_WORK_DIR)

    def test_run_with_dist_eval_hook(self):
        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        cfg = dict(
            model=model,
            optimizer=optimizer,
            batch_processor=None,
            lr_config=dict(policy='step', step=[2, 3]),
            optimizer_config=dict(grad_clip=None),
            checkpoint_config=None,
            log_config=dict(interval=4, hooks=[dict(type='TextLoggerHook')]),
            add_eval_hook=True
        )
        estimator = get_estimator(runner_creator, cfg, workers_per_node=2)
        epoch_stats = estimator.run([train_dataloader_creator], [('train', 1)])
        self.assertEqual(len(epoch_stats), MAX_EPOCH)

        if os.path.exists(TEMP_WORK_DIR):
            shutil.rmtree(TEMP_WORK_DIR)

    def test_run_with_batch_processor(self):
        model = Model2()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        cfg = dict(
            model=model,
            optimizer=optimizer,
            batch_processor=batch_processor,
            lr_config=dict(policy='step', step=[2, 3]),
            optimizer_config=dict(grad_clip=None),
            checkpoint_config=None,
            log_config=dict(interval=4, hooks=[dict(type='TextLoggerHook')])
        )
        estimator = get_estimator(runner_creator, cfg)
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

        if os.path.exists(TEMP_WORK_DIR):
            shutil.rmtree(TEMP_WORK_DIR)

    def test_save_load_ckpt(self):
        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        cfg = dict(
            model=model,
            optimizer=optimizer,
            batch_processor=None,
            lr_config=dict(policy='step', step=[2, 3]),
            optimizer_config=dict(grad_clip=None),
            checkpoint_config=dict(interval=1),
            log_config=dict(interval=4, hooks=[dict(type='TextLoggerHook')])
        )
        estimator = get_estimator(runner_creator, cfg)
        estimator.run([train_dataloader_creator], [('train', 1)])
        assert os.path.exists(os.path.join(TEMP_WORK_DIR, "epoch_1.pth"))
        assert os.path.exists(os.path.join(TEMP_WORK_DIR, "epoch_2.pth"))
        assert os.path.exists(os.path.join(TEMP_WORK_DIR, "epoch_3.pth"))
        assert os.path.exists(os.path.join(TEMP_WORK_DIR, "epoch_4.pth"))

        estimator.load_checkpoint(os.path.join(TEMP_WORK_DIR, "epoch_4.pth"))

        if os.path.exists(TEMP_WORK_DIR):
            shutil.rmtree(TEMP_WORK_DIR)

    def test_get_model(self):
        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        cfg = dict(
            model=model,
            optimizer=optimizer,
            batch_processor=None,
            lr_config=dict(policy='step', step=[2, 3]),
            optimizer_config=dict(grad_clip=None),
            checkpoint_config=None,
            log_config=dict(interval=4, hooks=[dict(type='TextLoggerHook')])
        )
        estimator = get_estimator(runner_creator, cfg)
        estimator.run([train_dataloader_creator], [('train', 1)])

        model_state_dict = estimator.get_model()
        assert model_state_dict


if __name__ == "__main__":
    pytest.main([__file__])

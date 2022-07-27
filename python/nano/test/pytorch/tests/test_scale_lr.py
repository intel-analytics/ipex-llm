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

import pytest
import math
from unittest import TestCase

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import create_test_data_loader
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch import nn
import torch.nn.functional as F

num_classes = 10
batch_size = 32
dataset_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class ResNetWith2Optimzers(pl.LightningModule):
    def __init__(self, learning_rate1=0.01, learning_rate2=0.05) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.backbone = vision.resnet18(pretrained=False, include_top=False, freeze=False)
        output_size = self.backbone.get_output_size()
        self.head = nn.Linear(output_size, num_classes)

    def on_train_epoch_start(self) -> None:
        world_size = self.trainer.strategy.world_size
        for lr_scheduler, lr in zip(self.lr_schedulers(), self.hparams):
            if lr_scheduler.last_epoch > self.trainer.strategy.auto_lr:
                assert math.isclose(lr_scheduler.get_last_lr()[0], 
                                    lr_scheduler.base_lrs[0])
            else:
                diff = lr_scheduler.base_lrs[0] * \
                    (1.0 - 1.0 / world_size) / \
                    (self.trainer.strategy.auto_lr)
                assert lr_scheduler.base_lrs[0] == self.hparams[lr] * world_size
                assert math.isclose(lr_scheduler.get_last_lr()[0],
                                    lr_scheduler.base_lrs[0] * 1.0 / world_size +
                                    lr_scheduler.last_epoch * diff,
                                    abs_tol=1e-7)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.nll_loss(logits, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        test_loss = F.nll_loss(logits, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer1 = torch.optim.SGD(
            self.backbone.parameters(),
            lr=self.hparams.learning_rate1
        )
        optimizer2 = torch.optim.Adam(
            self.head.parameters(),
            lr=self.hparams.learning_rate2
        )
        return [optimizer1, optimizer2]


class ResNetWithScheduler(pl.LightningModule):
    def __init__(self, learning_rate1=0.01, learning_rate2=0.02):
        super().__init__()

        self.save_hyperparameters()
        self.backbone = vision.resnet18(pretrained=False, include_top=False, freeze=False)
        output_size = self.backbone.get_output_size()
        self.head = nn.Linear(output_size, num_classes)

    def on_train_start(self):
        world_size = self.trainer.strategy.world_size
        for opt, lr_sch, lr in zip(self.optimizers(), self.lr_schedulers(), self.hparams):
            if hasattr(lr_sch, 'start_factor'):
                assert opt.param_groups[0]['lr'] == lr_sch.base_lrs[0] * lr_sch.start_factor
                return
            assert lr_sch.base_lrs[0] == self.hparams[lr] * world_size

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.nll_loss(logits, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        test_loss = F.nll_loss(logits, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer1 = torch.optim.SGD(
            self.backbone.parameters(),
            lr=self.hparams.learning_rate1
        )
        optimizer2 = torch.optim.Adam(
            self.head.parameters(),
            lr=self.hparams.learning_rate2
        )
        lr_scheduler1 = torch.optim.lr_scheduler.StepLR(
            optimizer1, step_size=1, gamma=0.5
        )
        if TORCH_VERSION_LESS_1_10:
            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
                optimizer2, step_size=1, gamma=0.1
            )
        else:
            lr_scheduler2 = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer2, start_factor=0.5
            )
        return (
            {"optimizer": optimizer1, "lr_scheduler": lr_scheduler1},
            {"optimizer": optimizer2, "lr_scheduler": lr_scheduler2}
        )


class TestScaleLr(TestCase):
    data_loader = create_data_loader(data_dir, batch_size, num_workers,
                                     data_transform, subset=dataset_size)
    test_data_loader = create_test_data_loader(data_dir, batch_size, num_workers,
                                               data_transform, subset=dataset_size)

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(
            os.path.join(os.path.join(os.path.join(test_dir, ".."), ".."), "..")
        )
        os.environ['PYTHONPATH'] = project_test_dir

    def test_scale_lr_subprocess(self):
        model = ResNetWithScheduler()
        trainer = Trainer(num_processes=2,
                          distributed_backend="subprocess",
                          auto_lr=True,
                          max_epochs=2,
                          callbacks=[LearningRateMonitor(logging_interval='epoch')])
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

    def test_scale_lr_spawn(self):
        model = ResNetWithScheduler()
        trainer = Trainer(num_processes=2,
                          distributed_backend='spawn',
                          auto_lr=True,
                          max_epochs=2,
                          callbacks=[LearningRateMonitor(logging_interval='epoch')]
                          )
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

    def test_warmup_subprocess(self):
        model = ResNetWith2Optimzers()
        trainer = Trainer(num_processes=2,
                          distributed_backend='subprocess',
                          auto_lr=3,
                          max_epochs=4,
                          callbacks=[LearningRateMonitor(logging_interval='epoch')])
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

    def test_warmup_spawn(self):
        model = ResNetWith2Optimzers()
        trainer = Trainer(num_processes=4,
                          distributed_backend='spawn',
                          auto_lr=True,
                          max_epochs=5)
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)


if __name__ == '__main__':
    pytest.main([__file__])

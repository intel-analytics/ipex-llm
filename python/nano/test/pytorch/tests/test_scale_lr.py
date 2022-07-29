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


from gc import callbacks
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
            {"optimizer": optimizer1, "lr_scheduler": {
                "scheduler": lr_scheduler1,
            }},
            {"optimizer": optimizer2, "lr_scheduler": lr_scheduler2}
        )


class CheckLinearLRScaleCallback(pl.Callback):
    def __init__(self, num_processes, lrs) -> None:
        self.world_size = num_processes
        self.lrs = lrs

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for lr, opt, sch in zip(self.lrs, pl_module.optimizers(), pl_module.lr_schedulers()):
            assert sch.base_lrs[0] == lr * self.world_size


class CheckWarmupCallback(CheckLinearLRScaleCallback):
    def __init__(self, num_processes, lrs, max_epochs, steps_per_epoch, warmup_epochs=None):
        super().__init__(num_processes, lrs)
        if warmup_epochs:
            self.warmup_epochs = warmup_epochs
            self.warmup_on_epoch = True
        else:
            self.warmup_epochs = steps_per_epoch * max_epochs // 10
            self.warmup_on_epoch = False

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx, unused: int = 0) -> None:
        if self.warmup_on_epoch:
            return
        if batch_idx > self.warmup_epochs or pl_module.current_epoch > 0:
            for lr, opt in zip(self.lrs, pl_module.optimizers()):
                assert opt.param_groups[0]['lr'] == lr * self.world_size
        else:
            for lr, opt in zip(self.lrs, pl_module.optimizers()):
                diff = (1.0 - 1.0 / self.world_size) * batch_idx / self.warmup_epochs + 1.0 / self.world_size
                assert opt.param_groups[0]['lr'] == lr * self.world_size * diff

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not self.warmup_on_epoch:
            return
        if pl_module.current_epoch > self.warmup_epochs:
            for lr, opt in zip(self.lrs, pl_module.optimizers()):
                assert opt.param_groups[0]['lr'] == lr * self.world_size
        else:
            for lr, opt in zip(self.lrs, pl_module.optimizers()):
                diff = (1.0 - 1.0 / self.world_size) * pl_module.current_epoch / self.warmup_epochs + 1.0 / self.world_size
                assert opt.param_groups[0]['lr'] == lr * self.world_size * diff


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
                          callbacks=[CheckLinearLRScaleCallback(2, [0.01, 0.02])])
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

    def test_scale_lr_spawn(self):
        model = ResNetWithScheduler()
        trainer = Trainer(num_processes=4,
                          distributed_backend='spawn',
                          auto_lr=True,
                          max_epochs=2,
                          callbacks=[CheckLinearLRScaleCallback(4, [0.01, 0.02])]
                          )
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

    def test_warmup_subprocess(self):
        model = ResNetWith2Optimzers()
        trainer = Trainer(num_processes=2,
                          distributed_backend='subprocess',
                          auto_lr=True,
                          max_epochs=4,
                          callbacks=[CheckWarmupCallback(2, [0.01, 0.05], 4, 4)])
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

    def test_warmup_spawn(self):
        model = ResNetWith2Optimzers()
        trainer = Trainer(num_processes=4,
                          distributed_backend='spawn',
                          auto_lr={'warmup_epochs': 4},
                          max_epochs=10,
                          callbacks=[CheckWarmupCallback(4, [0.01, 0.05], 10, 4, 4)])
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)


if __name__ == '__main__':
    pytest.main([__file__])

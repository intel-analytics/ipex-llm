import os

import pytest
from unittest import TestCase

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import create_test_data_loader
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch import Trainer
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

class Resnet_2_2(pl.LightningModule):
    def __init__(self, learning_rate1=0.01, learning_rate2=0.02):
        super().__init__()

        self.save_hyperparameters()
        self.backbone = vision.resnet18(pretrained=False, include_top=False, freeze=False)
        output_size = self.backbone.get_output_size()
        self.head = nn.Linear(output_size, num_classes)

    def on_train_start(self):
        world_size = self.trainer.strategy.world_size
        for opt, lr_sch, lr in zip(self.trainer.optimizers, self.trainer.lr_schedulers, self.hparams):
            lr_sch = lr_sch['scheduler']
            assert lr_sch.base_lrs[0] == self.hparams[lr] * world_size
            if hasattr(lr_sch, 'factor'):
                assert opt.param_groups[0]['lr'] == lr_sch.base_lrs[0] * lr_sch.factor
            else:
                assert opt.param_groups[0]['lr'] == lr_sch.base_lrs[0] * lr_sch.start_factor

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
        lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer1,
            start_factor=0.25
        )
        lr_scheduler2 = torch.optim.lr_scheduler.ConstantLR(
            optimizer2,
            factor=0.5
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
        model = Resnet_2_2()
        trainer = Trainer(num_processes=2,
                          distributed_backend="subprocess",
                          scale_lr=True,
                          max_epochs=2,
                          callbacks=[LearningRateMonitor(logging_interval='epoch')])
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

    def test_scale_lr_spawn(self):
        model = Resnet_2_2()
        trainer = Trainer(num_processes=2,
                          distributed_backend='spawn',
                          scale_lr=True,
                          max_epochs=2,
                          callbacks=[LearningRateMonitor(logging_interval='epoch')]
                          )
        trainer.fit(model, train_dataloaders=self.data_loader,
                    val_dataloaders=self.test_data_loader)

if __name__ == '__main__':
    pytest.main([__file__])


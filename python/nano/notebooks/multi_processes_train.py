import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.vision import transforms

def prepare_data(data_path, batch_size, num_workers):
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization()
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar10_normalization()
        ]
    )
    cifar10_dm = CIFAR10DataModule(
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms
    )
    return cifar10_dm

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, learning_rate=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

def multi_train():
    seed_everything(7)
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    BATCH_SIZE = 64
    NUM_WORKERS = int(os.cpu_count() / 2)
    data_module = prepare_data(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    model = LitResnet(learning_rate=0.05)
    model.datamodule = data_module
    multi_trainer = Trainer(num_processes=2,
                            use_ipex=False,
                            progress_bar_refresh_rate=10,
                            max_epochs=3,
                            logger=TensorBoardLogger("lightning_logs/", name="multi"),
                            callbacks=[LearningRateMonitor(logging_interval="step")])
    start = time()
    multi_trainer.fit(model, datamodule=data_module)
    mulit_fit_time = time() - start
    outputs = multi_trainer.test(model, datamodule=data_module)
    mulit_acc = outputs[0]['test_acc'] * 100
    return mulit_fit_time, mulit_acc

def ipex_train():
    seed_everything(7)
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    BATCH_SIZE = 64
    NUM_WORKERS = int(os.cpu_count() / 2)
    data_module = prepare_data(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    model = LitResnet(learning_rate=0.05)
    model.datamodule = data_module
    ipex_trainer = Trainer(num_processes=2,
                           use_ipex=True,
                           progress_bar_refresh_rate=10,
                           max_epochs=30,
                           logger=TensorBoardLogger("lightning_logs/", name="ipex"),
                           callbacks=[LearningRateMonitor(logging_interval="step")])
    start = time()
    ipex_trainer.fit(model, datamodule=data_module)
    ipex_fit_time = time() - start
    outputs = ipex_trainer.test(model, datamodule=data_module)
    ipex_acc = outputs[0]['test_acc'] * 100
    return ipex_fit_time, ipex_acc
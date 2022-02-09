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

# This file is adapted from PyTorch Lightning Tutorial.
# https://github.com/PyTorchLightning/lightning-tutorials/blob/main/
# lightning_examples/cifar10-baseline/baseline.py

# Copyright The PyTorch Lightning team.
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

# mypy: ignore-errors
import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import torchvision
from pytorch_lightning import LightningModule, seed_everything
from bigdl.nano.pytorch.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
import copy

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


seed_everything(7)

PATH_DATASETS = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 64
NUM_WORKERS = 0  # Multi-thread run sometimes raise quantization error


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
    pin_memory=False
)


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                            padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
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
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
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


model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm

trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=30,
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(model, cifar10_dm)

fp32_model = model
# FP32 testing
start = time()
outputs = trainer.test(fp32_model, datamodule=cifar10_dm)
time1 = time() - start
log_dict = outputs[0]
print("FP32 model size: %.3f MB" % fp32_model.model_size)
print("Throughput: %d s" % time1)
print("Accuracy: top1=%.2f%%" % (log_dict['test_acc'] * 100))

# Save FP32 checkpoint
trainer.save_checkpoint('ResNet18_on_CIFAR_FP32_CKPT.pth')

# Run post-training quantization
quantized = trainer.quantize(fp32_model, calib_dataloader=cifar10_dm.train_dataloader(),
                             val_dataloader=cifar10_dm.test_dataloader(),
                             metric=Accuracy(num_classes=10),
                             # recommend 'pytorch_fx' instead of 'pytorch' to enable automatic
                             # fusion
                             framework='pytorch_fx',
                             approach='static',
                             tuning_strategy='bayesian',
                             accuracy_criterion={
                                 'higher_is_better': True, 'relative': 0.1},
                             timeout=0,  # Limited time(seconds)
                             max_trials=10,  # Maximum traverse to get best model
                             raw_return=True)

# Wrap quantized model as a Pytorch-Lightning Module for testing
int8_model = copy.deepcopy(fp32_model)
int8_model.model = quantized

# Testing on quantized INT8 model
start = time()
outputs = trainer.test(int8_model, cifar10_dm)
time2 = time() - start
log_dict = outputs[0]
print("INT8 model size: %.3f MB" % int8_model.model_size)
print("Throughput: %d s" % time2)
print("Accuracy: top1=%.2f%%" % (log_dict['test_acc'] * 100))

# Save INT8 checkpoint
trainer.save_checkpoint('ResNet18_on_CIFAR_INT8_CKPT.pth')

print("After quantization, the model size reduces %d%%, throughput improves %d%%" % (
     (fp32_model.model_size - int8_model.model_size) / fp32_model.model_size * 100,
     (time1 - time2)/time1 * 100
))

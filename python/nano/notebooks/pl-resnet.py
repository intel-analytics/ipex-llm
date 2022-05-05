import os

import bigdl.nano.pytorch.vision.models
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
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.vision import transforms
from argparse import ArgumentParser
import intel_pytorch_extension as ipex

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
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitResnet")
        parser.add_argument("--learning_rate", type=float, default=0.05)
        return parent_parser

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

if __name__ == "__main__":
    seed_everything(7)
    parser = ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--data_path", type=str, default=os.environ.get("PATH_DATASETS", "."))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)
    parser = LitResnet.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args,
                                         progress_bar_refresh_rate=10,
                                         max_epochs=30,
                                         logger=TensorBoardLogger("lightning_logs/", name="nanoResnet"),
                                         callbacks=[LearningRateMonitor(logging_interval="step")])

    data_module = prepare_data(data_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers)
    model = LitResnet(vars(args)["learning_rate"])
    model.datamodule = data_module

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

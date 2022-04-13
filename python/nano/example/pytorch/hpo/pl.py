import os
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

from bigdl.nano.pytorch.trainer import Trainer
import bigdl.nano.automl.hpo as hpo


PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 1
DIR = os.getcwd()


class Net(nn.Module):
    # def __init__(self, dropout: float, output_dims: List[int]):
    def __init__(self, dropout, output_dim1, output_dim2):
        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = 28 * 28
        for output_dim in [output_dim1,output_dim2]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, CLASSES))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


@hpo.plmodel()
class LightningNet(pl.LightningModule):
    # def __init__(self, dropout: float, output_dims: List[int]):
    def __init__(self, dropout, output_dim1, output_dim2):
        super().__init__()
        self.model = Net(dropout, output_dim1, output_dim2)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.mnist_test = datasets.FashionMNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        mnist_full = datasets.FashionMNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )


def train():

    model = LightningNet(dropout=hpo.space.Real(0.3,0.5),
                         output_dim1=hpo.space.Categorical(16,32),
                         output_dim2=hpo.space.Categorical(16,32))
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
    trainer = Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
    )

    model = trainer.search(
        model,
        target_metric='val_acc',
        direction='maximize',
        datamodule = datamodule,
        n_trials=2,
        max_epochs=EPOCHS,
    )



if __name__ == "__main__":
    train()
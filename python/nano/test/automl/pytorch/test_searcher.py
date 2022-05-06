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

# Part of the test code is adapted from
# https://github.com/PyTorchLightning/pytorch-lightning/
# blob/master/pytorch_lightning/demos/boring_classes.py
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

import pytest
from unittest import TestCase

import torch

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule

from bigdl.nano.pytorch.trainer import Trainer
import bigdl.nano.automl.hpo.space as space
import bigdl.nano.automl.hpo as hpo

from bigdl.nano.automl.pytorch import HPOSearcher

class RandomDataset(Dataset):
    """
    Random Dataset.

    This class is modified from RandomDataset in
    https://github.com/PyTorchLightning/pytorch-lightning/
    blob/master/pytorch_lightning/demos/boring_classes.py

    :param _type_ Dataset: _description_
    """

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return self.len

@hpo.plmodel()
class CustomModel(LightningModule):
    """
    TestModel.

    This model is adapted from BoringModel in
    https://github.com/PyTorchLightning/pytorch-lightning/
    blob/master/pytorch_lightning/demos/boring_classes.py
    """

    def __init__(self,
                out_dim1,
                out_dim2,
                dropout_1,
                dropout_2):

        super().__init__()
        layers = []
        input_dim = 32
        for out_dim, dropout in [(out_dim1, dropout_1),
                                (out_dim2,dropout_2)]:
            layers.append(torch.nn.Linear(input_dim, out_dim))
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(dropout))
            input_dim = out_dim

        layers.append(torch.nn.Linear(input_dim, 2))

        self.layers: torch.nn.Module = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        self.log("val_loss", loss)
        # self.log("hp_metric", accuracy, on_step=False, on_epoch=True)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.layers.parameters(),lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class TestHPOSearcher(TestCase):

    def test_dummy_model(self):

        model = CustomModel(
            out_dim1=space.Categorical(16,32),
            out_dim2=space.Categorical(16,32),
            dropout_1=space.Real(0.1,0.5),
            dropout_2 = 0.2)

        trainer = Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=10,
        )
        searcher = HPOSearcher(trainer)
        searcher.search(
            model,
            target_metric='val_loss',
            direction='minimize',
            n_trials=2,
            max_epochs=2,
        )
        study = searcher.search_summary()
        assert(study)
        assert(study.best_trial)
        # score = trainer.callback_metrics["val_loss"].item()
        # print("final val_loss is:", score)



if __name__ == '__main__':
    pytest.main([__file__])
#
# Copyright 2018 Analytics Zoo Authors.
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
from unittest import TestCase

import numpy as np
import pytest

import torch
import torch.nn as nn
from zoo.orca.learn.pytorch import PyTorchTrainer

np.random.seed(1337)  # for reproducibility


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, a, b, size=1000):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


def get_data_loaders(config):
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 1000))
    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))
    return train_loader, validation_loader


def get_model(config):
    return nn.Linear(1, config.get("hidden_size", 1))


def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


class TestPyTorchTrainer(TestCase):
    def test_linear(self):
        trainer = PyTorchTrainer(model_creator=get_model,
                                 data_creator=get_data_loaders,
                                 optimizer_creator=get_optimizer,
                                 loss_creator=nn.MSELoss,
                                 config={"lr": 1e-2, "hidden_size": 1,
                                         "batch_size": 128})
        stats = trainer.train(nb_epoch=2)
        print(stats)


if __name__ == "__main__":
    pytest.main([__file__])

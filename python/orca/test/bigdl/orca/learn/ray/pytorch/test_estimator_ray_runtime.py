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
from unittest import TestCase

import torch
import torch.nn as nn

from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator


class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, nested_input=False):
        self.nested_input = nested_input
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.x = torch.cat([X1, X2], dim=0)
        Y1 = torch.zeros(size // 2, 1)
        Y2 = torch.ones(size // 2, 1)
        self.y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        if self.nested_input:
            return {'x':self.x[index, None]}, self.y[index, None]
        else:
            return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)

class SingleListDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, nested_input=True) -> None:
        super().__init__()
        self.size = size
        self.nested_input = nested_input
        X1_1 = torch.rand(self.size // 2, 1)
        X1_2 = torch.rand(self.size // 2, 1) + 1.5
        self.X1 = torch.cat([X1_1, X1_2], dim=0)

        X2_1 = torch.rand(self.size // 2, 1) + 1.5
        X2_2 = torch.rand(self.size // 2, 1) + 3.0
        self.X2 = torch.cat([X2_1, X2_2], dim=0)

        Y1 = torch.zeros(self.size // 2, 1)
        Y2 = torch.ones(self.size // 2, 1)
        self.Y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        if self.nested_input:
            return [self.X1[index], self.X2[index]], self.Y[index]
        else:
            return self.X1[index], self.X2[index], self.Y[index]
    
    def __len__(self):
        return self.size

class ComplicatedInputDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, nested_input=True) -> None:
        super().__init__()
        self.size = size
        X1_1 = torch.rand(self.size // 2, 1)
        X1_2 = torch.rand(self.size // 2, 1) + 1.5
        self.X1 = torch.cat([X1_1, X1_2], dim=0)

        X2_1 = torch.rand(self.size // 2, 1) + 1.5
        X2_2 = torch.rand(self.size // 2, 1) + 3.0
        self.X2 = torch.cat([X2_1, X2_2], dim=0)

        X3_1 = torch.rand(self.size // 2, 1) + 3.0
        X3_2 = torch.rand(self.size // 2, 1) + 4.5
        self.X3 = torch.cat([X3_1, X3_2], dim=0)

        X4_1 = torch.rand(self.size // 2, 1) + 4.5
        X4_2 = torch.rand(self.size // 2, 1) + 6.0
        self.X4 = torch.cat([X4_1, X4_2], dim=0)

        Y1 = torch.zeros(self.size // 2, 1)
        Y2 = torch.ones(self.size // 2, 1)
        self.Y = torch.cat([Y1, Y2], dim=0)
    
    def __getitem__(self, index):
        return (self.X1[index], self.X2[index]), {'x3': self.X3[index]}, self.X4[index], self.Y[index]
    
    def __len__(self):
        return self.size

DataSetMap = {"LinearDataset": LinearDataset,
              "SingleListDataset": SingleListDataset,
              "ComplicatedInputDataset": ComplicatedInputDataset}

def train_data_loader(config, batch_size):
    train_dataset = DataSetMap[config.get("dataset", "LinearDataset")](size=config.get("data_size", 1000),
                                                                       nested_input=config.get("nested_input", False))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size
    )
    return train_loader

def val_data_loader(config, batch_size):
    val_dataset = DataSetMap[config.get("dataset", "LinearDataset")](size=config.get("val_size", 400),
                                                                     nested_input=config.get("nested_input", False))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    return validation_loader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

class DictInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_['x'])
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

class SingleListInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_list):
        x = torch.cat(input_list, dim=1)
        x = self.fc(x)
        x = self.out_act(x)
        return x

class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = self.out_act(x)
        return x

class ComplicatedInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.out_act = nn.Sigmoid()
    
    def forward(self, x1_x2, x3_dict, x4):
        x = torch.cat((x1_x2[0], x1_x2[1], x3_dict['x3'], x4), dim=1)
        x = self.fc(x)
        x = self.out_act(x)
        return x


ModelMap = {"Net": Net,
            "SingleListInputModel": SingleListInputModel,
            "MultiInputModel": MultiInputModel,
            "DictInputNet": DictInputNet,
            "ComplicatedInputModel": ComplicatedInputModel}

def get_model(config):
    torch.manual_seed(0)
    return ModelMap[config.get("model", "Net")]()

def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))

class TestPytorchEstimator(TestCase):
    def setUp(self):
        init_orca_context(runtime="ray", address="localhost:6379")

    def tearDown(self):
        stop_orca_context()

    def test_train(self):
        estimator = Estimator.from_torch(model=get_model,
                                        optimizer=get_optimizer,
                                        loss=nn.BCELoss(),
                                        metrics=Accuracy(),
                                        config={"lr": 1e-2},
                                        workers_per_node=2,
                                        backend="ray",
                                        sync_stats=True)

        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)

        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)

        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)

        assert 0 < end_val_stats["Accuracy"] < 1
        assert estimator.get_model()

        # sanity check that training worked
        dloss = end_val_stats["val_loss"] - start_val_stats["val_loss"]
        dacc = (end_val_stats["Accuracy"] - start_val_stats["Accuracy"])
        print(f"dLoss: {dloss}, dAcc: {dacc}")
        assert dloss < 0 < dacc, "training sanity check failed. loss increased!"
    
    def test_singlelist_input(self):
        estimator = Estimator.from_torch(model=get_model,
                                        optimizer=get_optimizer,
                                        loss=nn.BCELoss(),
                                        metrics=Accuracy(),
                                        config={"lr": 1e-2,
                                                "model": "SingleListInputModel",
                                                "dataset": "SingleListDataset",
                                                "nested_input": True},
                                        workers_per_node=2,
                                        backend="ray",
                                        sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)

        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)

        assert 0 < end_val_stats["Accuracy"] < 1
        assert estimator.get_model()

        # sanity check that training worked
        dloss = end_val_stats["val_loss"] - start_val_stats["val_loss"]
        dacc = (end_val_stats["Accuracy"] - start_val_stats["Accuracy"])
        print(f"dLoss: {dloss}, dAcc: {dacc}")
        assert dloss < 0 < dacc, "training sanity check failed. loss increased!"

    def test_multi_input(self):
        estimator = Estimator.from_torch(model=get_model,
                                        optimizer=get_optimizer,
                                        loss=nn.BCELoss(),
                                        metrics=Accuracy(),
                                        config={"lr": 1e-2,
                                                "model": "MultiInputModel",
                                                "dataset": "SingleListDataset",
                                                "nested_input": False},
                                        workers_per_node=2,
                                        backend="ray",
                                        sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)

        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)

        assert 0 < end_val_stats["Accuracy"] < 1
        assert estimator.get_model()

        # sanity check that training worked
        dloss = end_val_stats["val_loss"] - start_val_stats["val_loss"]
        dacc = (end_val_stats["Accuracy"] - start_val_stats["Accuracy"])
        print(f"dLoss: {dloss}, dAcc: {dacc}")
        assert dloss < 0 < dacc, "training sanity check failed. loss increased!"

    def test_dict_input(self):
        estimator = Estimator.from_torch(model=get_model,
                                        optimizer=get_optimizer,
                                        loss=nn.BCELoss(),
                                        metrics=Accuracy(),
                                        config={"lr": 1e-2,
                                                "model": "DictInputNet",
                                                "dataset": "LinearDataset",
                                                "nested_input": True},
                                        workers_per_node=2,
                                        backend="ray",
                                        sync_stats=True)

        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)

        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)

        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)

        assert 0 < end_val_stats["Accuracy"] < 1
        assert estimator.get_model()

        # sanity check that training worked
        dloss = end_val_stats["val_loss"] - start_val_stats["val_loss"]
        dacc = (end_val_stats["Accuracy"] - start_val_stats["Accuracy"])
        print(f"dLoss: {dloss}, dAcc: {dacc}")
        assert dloss < 0 < dacc, "training sanity check failed. loss increased!"

    def test_complicated_input(self):
        estimator = Estimator.from_torch(model=get_model,
                                        optimizer=get_optimizer,
                                        loss=nn.BCELoss(),
                                        metrics=Accuracy(),
                                        config={"lr": 1e-2,
                                                "model": "ComplicatedInputModel",
                                                "dataset": "ComplicatedInputDataset"},
                                        workers_per_node=2,
                                        backend="ray",
                                        sync_stats=True)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(start_val_stats)
        
        train_stats = estimator.fit(train_data_loader, epochs=1, batch_size=32)
        print(train_stats)

        end_val_stats = estimator.evaluate(val_data_loader, batch_size=32)
        print(end_val_stats)

        assert 0 < end_val_stats["Accuracy"] < 1
        assert estimator.get_model()

        # sanity check that training worked
        dloss = end_val_stats["val_loss"] - start_val_stats["val_loss"]
        dacc = (end_val_stats["Accuracy"] - start_val_stats["Accuracy"])
        print(f"dLoss: {dloss}, dAcc: {dacc}")
        assert dloss < 0 < dacc, "training sanity check failed. loss increased!"


if __name__ == "__main__":
    pytest.main([__file__])

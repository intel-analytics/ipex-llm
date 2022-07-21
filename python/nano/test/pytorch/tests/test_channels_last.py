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
import torch
import torchmetrics
from unittest import TestCase
from bigdl.nano.pytorch import Trainer
from torchvision.models.resnet import ResNet, BasicBlock
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_12
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import create_test_data_loader

num_classes = 10
batch_size = 32
dataset_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")

class customResNet(ResNet):
    def __init__(self):
        super(customResNet, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.head = torch.nn.Linear(self.fc.out_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if not TORCH_VERSION_LESS_1_12:
            assert x.is_contiguous(memory_format=torch.channels_last)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.head(x)

        return x

class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, (1, 2), bias=False)
        self.conv1.weight.data.fill_(1.0)
    
    def forward(self, input):
        x = self.conv1(input)
        if not TORCH_VERSION_LESS_1_12:
            assert x.is_contiguous(memory_format=torch.channels_last)
        output = torch.flatten(x, 1)
        return output

class TestChannelsLast(TestCase):
    model = customResNet()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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

    def test_trainer_channels_last(self):
        pl_model = LightningModuleFromTorch(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(max_epochs=1, channels_last=True)
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.test_data_loader)

    def test_trainer_channels_last_correctness(self):
        # dataset: features: [[[[1, 0]] [[1, 0]] [[0, 3]] [[1, 1]]    
        #                      [[1, 0]] [[2, 0]] [[1, 0]] [[2, 1]]]]    feature.shape=(4,2,1,2)
        # dataset: labels: [[0], [1], [0], [1]]                         label.shape=(4,1)
        # model: y = I * C = I1 * C1 + I2 * C2 = I11 * C11 + I12 * C12 + I21 * C21 + I22 * C22
        # loss = (y - label) ^ 2
        # dloss/dC11 = 2 * (y - label) * dy/dC11    dloss/dC12 = 2 * (y - label) * dy/dC12
        # dloss/dC21 = 2 * (y - label) * dy/dC21    dloss/dC22 = 2 * (y - label) * dy/dC22
        # end of iteration:
        #    avg_grad_c11 = avg([4, 4, 0, 8])   = 4
        #    avg_grad_c12 = avg([0, 0, 24, 8])  = 8
        #    avg_grad_c21 = avg([4, 8, 8, 16])  = 9
        #    avg_grad_c22 = avg([0, 0, 0, 8])   = 2
        #    weight_C = 1 - 0.25 * avg_grad_C   =
        #    [[[[1, 1]]         [[[[1,      2]]         [[[[0,     -1]]
        #      [[1, 1]]]]   -     [[2.25, 0.5]]]]   =     [[-1.25 0.5]]]]
        #
        #
        #
        model = ConvModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        loss = torch.nn.MSELoss()
        pl_module = Trainer.compile(model=model, loss=loss, optimizer=optimizer)
        trainer = Trainer(max_epochs=1, channels_last=True)

        x = torch.Tensor([
            [[[1, 0]], [[1, 0]]],
            [[[1, 0]], [[2, 0]]],
            [[[0, 3]], [[1, 0]]],
            [[[1, 1]], [[2, 1]]]
                            ])
        y = torch.Tensor([[0.0], [1.0], [0.0], [1.0]])
        dataset = torch.utils.data.TensorDataset(x, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        trainer.fit(pl_module, data_loader)
        result = torch.tensor([[[[0.0, -1.0]], [[-1.25, 0.5]]]])
        assert pl_module.model.conv1.weight.equal(result)

    def test_trainer_channels_last_correctness_subprocess(self):
        model = ConvModel()
        model.conv1 = torch.nn.Conv2d(2, 1, (1, 2), bias=False)
        model.conv1.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        loss = torch.nn.MSELoss()
        pl_module = Trainer.compile(model=model, loss=loss, optimizer=optimizer)
        trainer = Trainer(max_epochs=1,
                          channels_last=True,
                          distributed_backend="subprocess",
                          num_processes=2)

        x = torch.Tensor([
            [[[1, 0]], [[1, 0]]],
            [[[1, 0]], [[2, 0]]],
            [[[0, 3]], [[1, 0]]],
            [[[1, 1]], [[2, 1]]]
                            ])
        y = torch.Tensor([[0], [1], [0], [1]])
        dataset = torch.utils.data.TensorDataset(x, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        trainer.fit(pl_module, data_loader)
        result = torch.tensor([[[[0.0, -1.0]], [[-1.25, 0.5]]]])
        assert pl_module.model.conv1.weight.equal(result)

    def test_trainer_channels_last_subprocess(self):
        pl_model = LightningModuleFromTorch(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(max_epochs=1,
                          num_processes=2,
                          distributed_backend="subprocess", 
                          channels_last=True)
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.test_data_loader)


class TestChannelsLastSpawn(TestCase):
    model = customResNet()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_loader = create_data_loader(data_dir, batch_size, num_workers,
                                        data_transform, subset=dataset_size)
    test_data_loader = create_test_data_loader(data_dir, batch_size, num_workers,
                                                data_transform, subset=dataset_size)

    def test_trainer_channels_last_spawn(self):
        pl_model = LightningModuleFromTorch(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(max_epochs=1,
                          num_processes=2,
                          distributed_backend="spawn", 
                          channels_last=True)
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.data_loader)

    def test_trainer_channels_last_correctness_spawn(self):
        model = ConvModel()
        model.conv1 = torch.nn.Conv2d(2, 1, (1, 2), bias=False)
        model.conv1.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        loss = torch.nn.MSELoss()
        pl_module = Trainer.compile(model=model, loss=loss, optimizer=optimizer)
        trainer = Trainer(max_epochs=1,
                          channels_last=True,
                          distributed_backend="spawn",
                          num_processes=2)

        x = torch.Tensor([
            [[[1, 0]], [[1, 0]]],
            [[[1, 0]], [[2, 0]]],
            [[[0, 3]], [[1, 0]]],
            [[[1, 1]], [[2, 1]]]
                            ])
        y = torch.Tensor([[0], [1], [0], [1]])
        dataset = torch.utils.data.TensorDataset(x, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        trainer.fit(pl_module, data_loader)
        result = torch.tensor([[[[0.0, -1.0]], [[-1.25, 0.5]]]])
        assert pl_module.model.conv1.weight.equal(result)

    
if __name__ == '__main__':
    pytest.main([__file__])

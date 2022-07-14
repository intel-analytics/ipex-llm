import os
from turtle import forward
import pytest
import torch
import torchmetrics
from unittest import TestCase
from bigdl.nano.pytorch import Trainer
from torchvision.models.resnet import ResNet, BasicBlock
from torch.utils.data import DataLoader, TensorDataset
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_12
from test.pytorch.tests.test_lightning import ResNet18
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

class TestChannelsLast(TestCase):
    model = customResNet()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dummy_input = torch.rand(8, 3, 128, 128)
    dummy_label = torch.randint(1, 10, (8,))

    dataset = TensorDataset(dummy_input, dummy_label)
    train_loader = DataLoader(dataset)

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
        trainer.fit(pl_model, self.train_loader)
        trainer.test(pl_model, self.train_loader)

    def test_trainer_channels_last_subprocess(self):
        pl_model = LightningModuleFromTorch(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(max_epochs=1,
                          num_processes=2,
                          distributed_backend="subprocess", 
                          channels_last=True)
        trainer.fit(pl_model, self.train_loader)
        trainer.test(pl_model, self.train_loader)


class TestChannelsLastSpawn(TestCase):
    model = customResNet()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dummy_input = torch.rand(8, 3, 128, 128)
    dummy_label = torch.randint(1, 10, (8,))

    dataset = TensorDataset(dummy_input, dummy_label)
    train_loader = DataLoader(dataset)

    def test_trainer_channels_last_spaw(self):
        pl_model = LightningModuleFromTorch(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(max_epochs=1,
                          num_processes=2,
                          distributed_backend="spawn", 
                          channels_last=True)
        trainer.fit(pl_model, self.train_loader)
        trainer.test(pl_model, self.train_loader)

    
if __name__ == '__main__':
    pytest.main([__file__])

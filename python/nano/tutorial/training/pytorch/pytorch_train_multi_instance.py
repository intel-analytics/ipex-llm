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

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import OxfordIIITPet
from torchmetrics import Accuracy

from pytorch_lightning import seed_everything
from bigdl.nano.pytorch.torch_nano import TorchNano


class MyPytorchModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 37.
        self.model.fc = nn.Linear(num_ftrs, 37)

    def forward(self, x):
        return self.model(x)


def create_dataloaders():
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=.5, hue=.3),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    # Apply data augmentation to the tarin_dataset
    train_dataset = OxfordIIITPet(root="/tmp/data", transform=train_transform, download=True)
    val_dataset = OxfordIIITPet(root="/tmp/data", transform=val_transform)

    # obtain training indices that will be used for validation
    indices = torch.randperm(len(train_dataset))
    val_size = len(train_dataset) // 4
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])

    # prepare data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    return train_dataloader, val_dataloader


# subclass TorchNano and override its train() method
class MyNano(TorchNano):
    # move the body of your existing train function into TorchNano train method
    def train(self):
        seed_everything(42)
        model = MyPytorchModule()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        loss_fuc = torch.nn.CrossEntropyLoss()
        train_loader, val_loader = create_dataloaders()

        # call `setup` to prepare for model, optimizer(s) and dataloader(s) for accelerated training
        model, optimizer, (train_loader, val_loader) = self.setup(model, optimizer,
                                                                  train_loader, val_loader)
        num_epochs = 5

        # EPOCH LOOP
        for epoch in range(num_epochs):

            # TRAINING LOOP
            model.train()
            train_loss, num = 0, 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fuc(output, target)
                # replace the loss.backward() with self.backward(loss)
                self.backward(loss)
                optimizer.step()

                train_loss += loss.sum()
                num += 1
            print(f'Train Epoch: {epoch}, loss: {train_loss/num}')


if __name__ == '__main__':
    # Multi-instance Training
    #
    # It is often beneficial to use multiple instances
    # for training if a server contains multiple sockets or
    # many cores, so that the workload can make full use of
    # all CPU cores.
    #
    # When using data-parallel training, the batch size is equivalent to
    # becoming n times larger, where n is the number of parallel processes.
    # We should to scale the learning rate to n times as well to achieve the
    # same effect as single instance training.
    # However, scaling the learning rate linearly may lead to poor convergence
    # at the beginning of training, so we should gradually increase the
    # learning rate to n times, and this is called 'learning rate warmup'.
    # 
    # Fortunately, BigDL-Nano makes it very easy to conduct multi-instance 
    # training correctly. It will handle all these for you.
    #
    # In BigDL-Nano, you can simply set the num_processes in
    # TorchNano to enable multi-instance training. In addition, it will automatically
    # apply learning rate scaling and warmup for your training.
    #
    MyNano(num_processes=2).train()

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

import torch
import torchmetrics
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import Callback
from typing import Any

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.algorithms.selective_backprop import SelectiveBackprop

num_classes = 10
batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class CheckBatchSize(Callback):

    def __init__(self,
                 start: float = 0.5,
                 keep: float = 0.5,
                 end: float = 0.9,
                 interrupt: int = 2,
                 batch_size: int = 256):
        self.start = start
        self.keep = keep
        self.end = end
        self.interrupt = interrupt
        self.batch_size = batch_size

    def __should_selective_backprop(
        self,
        current_duration: float,
        batch_idx: int,
        start: float = 0.5,
        end: float = 0.9,
        interrupt: int = 2,
    ) -> bool:
        is_interval = ((current_duration >= start)
                       and (current_duration < end))
        is_step = ((interrupt == 0) or ((batch_idx + 1) % interrupt != 0))

        return is_interval and is_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch: Any,
                           batch_idx: int, dataloader_idx: int) -> None:
        elapsed_duration = float(trainer.current_epoch) / \
            float(trainer.max_epochs)
        if self.__should_selective_backprop(elapsed_duration, batch_idx,
                                            self.start, self.end,
                                            self.interrupt):
            current_batch_size = len(batch[1])
            ideal_batch_size = int(self.keep * self.batch_size)
            assert current_batch_size == ideal_batch_size, \
                'Batch size is not right. Selective_backprop may not work.'


class ResNet18(nn.Module):

    def __init__(self, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained,
                                   include_top=include_top,
                                   freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = torch.nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


def create_data_loader(dir, batch_size, num_workers, transform, subset=50, shuffle=True, sampler=False):
    train_set = CIFAR10(root=dir, train=True,
                        download=True, transform=transform)
    # `subset` is the number of subsets. The larger the number, the smaller the training set.
    mask = list(range(0, len(train_set), subset))
    train_subset = torch.utils.data.Subset(train_set, mask)
    if sampler:
        sampler_set = SequentialSampler(train_subset)
        data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, sampler=sampler_set)
    else:
        data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])


model = ResNet18(pretrained=False, include_top=False, freeze=True)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def main():
    pl_model = LightningModuleFromTorch(
        model,
        loss,
        optimizer,
        metrics=[
            torchmetrics.F1(num_classes),
            torchmetrics.Accuracy(num_classes=10)
        ])
    data_loader = create_data_loader(data_dir, batch_size, num_workers,
                                        data_transform)
    # get the loss function without reduction
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    # pass proper arguments to selective backprop
    sb = SelectiveBackprop(start=0.5,
                            keep=0.5,
                            end=0.9,
                            scale_factor=1,
                            interrupt=2,
                            loss_fn=loss_fn)
    # check if selective backprop works
    batch_size_check = CheckBatchSize(start=0.5,
                                        keep=0.5,
                                        end=0.9,
                                        interrupt=2,
                                        batch_size=batch_size)
    # pass the algorithm by algorithms=[sb,]
    trainer = Trainer(max_epochs=10,
                        log_every_n_steps=1,
                        algorithms=[sb],
                        callbacks=[batch_size_check])
    trainer.fit(pl_model, data_loader, data_loader)


if __name__ == '__main__':
    main()

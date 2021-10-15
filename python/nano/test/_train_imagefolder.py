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
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision.transforms import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from bigdl.nano.pytorch.vision.models import ImageClassifier


class Net(ImageClassifier):
    # Common case: fully-connected top layer

    def __init__(self, backbone):
        super().__init__(backbone=backbone, num_classes=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, amsgrad=True)
        return optimizer


def create_data_loader(root_dir, batch_size):
    dir_path = os.path.realpath(root_dir)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    catdogs = ImageFolder(dir_path, data_transform)
    val_num = len(catdogs) // 10
    train_num = len(catdogs) - val_num
    train_set, val_set = torch.utils.data.random_split(
        catdogs, [train_num, val_num])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_torch_lightning(model, root_dir, batch_size):
    train_loader, val_loader = create_data_loader(root_dir, batch_size)
    net = Net(model)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(net, train_loader)
    trainer.test(net, val_loader)
    print('pass')

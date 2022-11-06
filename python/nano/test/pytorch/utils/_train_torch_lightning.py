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


import torch
from copy import deepcopy
import pytorch_lightning as pl

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from bigdl.nano.pytorch.vision.models import ImageClassifier

num_classes = 10

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

test_data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.Resize(128),
    transforms.ToTensor()
])


class Net(ImageClassifier):
    # Common case: fully-connected top layer

    def __init__(self, backbone):
        super().__init__(backbone=backbone, num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, amsgrad=True)
        return optimizer


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


def create_test_data_loader(dir, batch_size, num_workers, transform, subset=50):
    '''
    This function is to create a fixed dataset without any randomness
    '''
    train_set = CIFAR10(root=dir, train=False,
                        download=True, transform=transform)
    # `subset` is the number of subsets. The larger the number, the smaller the training set.
    mask = list(range(0, len(train_set), subset))
    train_subset = torch.utils.data.Subset(train_set, mask)

    data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return data_loader


def train_with_linear_top_layer(model_without_top, batch_size, num_workers, data_dir,
                                use_ipex=False):
    model = Net(model_without_top)
    train_torch_lightning(model, batch_size, num_workers,
                          data_dir, use_ipex=use_ipex)


def train_torch_lightning(model, batch_size, num_workers, data_dir, use_ipex=False):
    orig_parameters = deepcopy(model.state_dict())
    # list to store the right key of dict
    orig_parameters_list = deepcopy(list(model.named_parameters()))

    train_loader = create_data_loader(
        data_dir, batch_size, num_workers, data_transform)

    from bigdl.nano.pytorch import Trainer
    trainer = Trainer(max_epochs=1, use_ipex=use_ipex)

    trainer.fit(model, train_loader)

    trained_parameters = model.state_dict()

    # Check if the training and the freeze operation is successful
    for i in range(len(orig_parameters_list)):
        name, para = orig_parameters_list[i]
        para1 = orig_parameters[name]
        para2 = trained_parameters[name]

        if name == "model.1.bias" or name == "model.1.weight" or \
                name == "new_classifier.1.bias" or name == "new_classifier.1.weight":
            # Top layer is trained
            if torch.all(torch.eq(para1, para2)):
                raise Exception("Parameter " + name + " remains the same after training.")
        else:
            # Frozen parameters should not change
            if not torch.all(torch.eq(para1, para2)):
                raise Exception(name + " freeze failed.")
    print("pass")

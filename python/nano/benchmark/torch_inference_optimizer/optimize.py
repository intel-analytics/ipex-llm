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
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch import InferenceOptimizer


def create_data_loader(dir, batch_size=1, subset=50, shuffle=True):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])
    train_set = CIFAR10(root=dir, train=True, download=True, transform=data_transform)
    # `subset` is the number of subsets. The larger the number, the smaller the training set.
    mask = list(range(0, len(train_set), subset))
    train_subset = torch.utils.data.Subset(train_set, mask)
    data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


def optimize():
    data_dir = "data"
    save_dir = "models"

    model = ResNet18(10)
    loader = create_data_loader(data_dir)
    opt = InferenceOptimizer()
    if "OMP_NUM_THREADS" in os.environ:
        thread_num = int(os.environ["OMP_NUM_THREADS"])
    else:
        thread_num = None

    opt.optimize(
        model=model,
        training_data=loader,
        thread_num=thread_num,
        search_mode="all"
    )

    os.makedirs(save_dir, exist_ok=True)
    options = list(InferenceOptimizer.ALL_INFERENCE_ACCELERATION_METHOD.keys())
    for option in options:
        try:
            model = opt.get_model(option)
            opt.save(model, os.path.join(save_dir, option))
        except Exception:
            pass
    

if __name__ == '__main__':
    optimize()

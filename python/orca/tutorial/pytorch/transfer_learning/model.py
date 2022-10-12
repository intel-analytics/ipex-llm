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
# ==============================================================================
# Most of the pytorch code is adapted from PyTorch's transfer learning tutorial for
# hymenoptera dataset.
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models


class ConvNetModel:
    def __init__(self):
        return

    # Create the model
    @staticmethod
    def model_creator(config):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(config['device'])
        return model

    # Create the optimizer
    @staticmethod
    def optimizer_creator(model, config):
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    @staticmethod
    def scheduler_creator(optimizer, config):
        return lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


class FixedConvNetModel:
    def __init__(self):
        return

    # Create the model
    @staticmethod
    def model_creator(config):
        model = torchvision.models.resnet18(pretrained=True)
        # Freeze all the network except the final layer.
        for param in model.parameters():
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(config['device'])
        return model

    # Create the optimizer
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    @staticmethod
    def optimizer_creator(model, config):
        return optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    @staticmethod
    def scheduler_creator(optimizer, config):
        return lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

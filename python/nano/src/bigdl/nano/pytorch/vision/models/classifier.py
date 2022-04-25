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


from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ._utils import BackboneModule


class ImageClassifier(pl.LightningModule):
    """A common LightningModule implementation of a image classifier."""

    def __init__(self, backbone: BackboneModule,
                 num_classes: int,
                 head: nn.Module = None,
                 criterion: nn.Module = None):
        """
        Create a ImageClassifier for common image classifier task.

        :param backbone: a backbone network of this ImageClassifer
        :param num_classes: the number of classification classes
        :param head: the head network on top of the backone network
        :param criterion: criterion for the classifier, default to CrossEntropyLoss
        """
        super().__init__()

        if head is None:
            output_size = backbone.get_output_size()
            head = nn.Linear(output_size, num_classes)
        self.model = torch.nn.Sequential(backbone, head)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        self.criterion = criterion

    def forward(self, x):
        """Run regular forward of a pytorch model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """A training step of a ImageClassifier."""
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        """A test step of a ImageClassifier."""
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(y == pred).item() / (len(y) * 1.0)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

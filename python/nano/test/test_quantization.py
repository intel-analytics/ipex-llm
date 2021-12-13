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
from unittest import TestCase

import torch
import torchmetrics
from test._train_torch_lightning import create_data_loader, data_transform
from torch import nn

from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.vision.models import vision

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
trainer = Trainer(max_epochs=1)
pl_model = Trainer.compile(model, loss, optimizer, metrics=[torchmetrics.F1(10)])
train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)


class TestQuantizationINC(TestCase):
    def test_trainer_quantize_ptq(self):
        quantized_model = trainer.quantize(pl_model, train_loader, metric='F1',
                                           framework='pytorch_fx', approach='ptsq',
                                           accuracy_criterion={'relative':         0.99,
                                                               'higher_is_better': True})
        if quantized_model:
            trainer.validate(quantized_model, train_loader)
            trainer.test(quantized_model, train_loader)

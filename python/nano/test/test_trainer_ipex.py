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

import pytest
import torch
import torchmetrics
from test._train_torch_lightning import create_data_loader, data_transform
from test._train_torch_lightning import train_with_linear_top_layer
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


class TestTrainer(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)

    def test_resnet18_ipex(self):
        resnet18 = vision.resnet18(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet18, batch_size, num_workers, data_dir,
            use_orca_lite_trainer=True)

    def test_trainer_compile(self):
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        trainer.fit(pl_model, self.train_loader)

    def test_trainer_quantize_inc_ptq(self):
        train_loader_iter = iter(self.train_loader)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer,
                                   metrics=[torchmetrics.F1(10)])

        # Case 1: Default
        qmodel = trainer.quantize(pl_model, self.train_loader)
        if qmodel:
            out = qmodel(next(train_loader_iter)[0])
            assert out.shape == torch.Size([256, 10])

        # Case 2: Override by arguments
        qmodel = trainer.quantize(pl_model, self.train_loader, self.train_loader,
                                  metric='F1', framework='pytorch_fx', approach='static',
                                  tuning_strategy='basic',
                                  accuracy_criterion={'relative':         0.99,
                                                      'higher_is_better': True})

        if qmodel:
            out = qmodel(next(train_loader_iter)[0])
            assert out.shape == torch.Size([256, 10])


if __name__ == '__main__':
    pytest.main([__file__])

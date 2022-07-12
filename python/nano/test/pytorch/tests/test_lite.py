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
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from torch import nn

from bigdl.nano.pytorch.utils import LIGHTNING_VERSION_LESS_1_6
if not LIGHTNING_VERSION_LESS_1_6:
    from bigdl.nano.pytorch.lite import LightningLite
    from bigdl.nano.pytorch.vision.models import vision

    batch_size = 256
    num_workers = 0
    data_dir = os.path.join(os.path.dirname(__file__), "../data")


    class ResNet18(nn.Module):
        def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
            super().__init__()
            backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
            output_size = backbone.get_output_size()
            head = nn.Linear(output_size, num_classes)
            self.model = nn.Sequential(backbone, head)

        def forward(self, x):
            return self.model(x)


    class Lite(LightningLite):
        def run(self):
            model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
            loss = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)

            model, optimizer = self.setup(model, optimizer)
            train_loader = self.setup_dataloaders(train_loader)
            model.train()

            max_epochs = 1
            for _i in range(max_epochs):
                total_loss, num = 0, 0
                for X, y in train_loader:
                    optimizer.zero_grad()
                    l = loss(model(X), y)
                    self.backward(l)
                    optimizer.step()
                    
                    total_loss += l.sum()
                    num += 1
                print(f'avg_loss: {total_loss / num}')


    class TestLite(TestCase):
        def test_lite(self):
            Lite().run()


if __name__ == '__main__':
    pytest.main([__file__])

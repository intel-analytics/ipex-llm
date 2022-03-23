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
from torch import nn

from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.trainer import Trainer

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.tests.test_trainer import ResNet18

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class TestPlugin(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(
            os.path.join(os.path.join(os.path.join(test_dir, ".."), ".."), "..")
        )
        os.environ['PYTHONPATH'] = project_test_dir

    def test_trainer_subprocess_plugin(self):
        pl_model = LightningModuleFromTorch(self.model, self.loss, self.optimizer)
        trainer = Trainer(num_processes=2, distributed_backend="subprocess", max_epochs=4)
        trainer.fit(pl_model, self.train_loader)
        trainer.test(pl_model, self.train_loader)


if __name__ == '__main__':
    pytest.main([__file__])

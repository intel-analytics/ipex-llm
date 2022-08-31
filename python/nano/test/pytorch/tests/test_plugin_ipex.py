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
import pytest
from unittest import TestCase

import torch
import torchmetrics
from torch import nn

from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.pytorch import Trainer
from bigdl.nano.common import check_avx512
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import create_test_data_loader
from test.pytorch.utils._train_ipex_callback import CheckIPEXCallback, CheckIPEXFusedStepCallback
from test.pytorch.tests.test_lightning import ResNet18

num_classes = 10
batch_size = 32
dataset_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class TestPlugin(TestCase):
    model = ResNet18(pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_loader = create_data_loader(data_dir, batch_size, num_workers,
                                     data_transform, subset=dataset_size)
    test_data_loader = create_test_data_loader(data_dir, batch_size, num_workers,
                                               data_transform, subset=dataset_size)

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(
            os.path.join(os.path.join(os.path.join(test_dir, ".."), ".."), "..")
        )
        os.environ['PYTHONPATH'] = project_test_dir

    def test_trainer_subprocess_plugin(self):
        pl_model = LightningModule(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(num_processes=2, distributed_backend="subprocess",
                          max_epochs=4, use_ipex=True,
                          callbacks=[CheckIPEXCallback()])
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.test_data_loader)

    def test_trainer_spawn_plugin_bf16(self):
        # IPEX BF16 weight prepack needs the cpu support avx512bw, avx512vl and avx512dq
        model = ResNet18(pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = LightningModule(
            model, loss, optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(num_processes=2, distributed_backend="spawn",
                          max_epochs=4, use_ipex=True, precision="bf16",
                          callbacks=[CheckIPEXCallback(), CheckIPEXFusedStepCallback()])
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.test_data_loader)
        if trainer.use_ipex and TORCH_VERSION_LESS_1_10:
            import intel_pytorch_extension as ipex
            # Diable IPEX AMP
            # Avoid affecting other tests
            ipex.enable_auto_mixed_precision(None)


if __name__ == '__main__':
    pytest.main([__file__])

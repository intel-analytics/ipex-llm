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
from torch.optim.lr_scheduler import OneCycleLR
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_ipex_callback import CheckIPEXFusedStepCallback
from torch import nn

from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.common import check_avx512

batch_size = 256
max_epochs = 2
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


class TestTrainer(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler_dict = {
        "scheduler": OneCycleLR(
            optimizer,
            0.1,
            epochs=max_epochs,
            steps_per_epoch=len(train_loader),
        ),
        "interval": "step",
    }

    def test_trainer_save_checkpoint(self):
        # `save_checkpoint` may report an error when using ipex 1.9 and custom lr_scheduler
        trainer = Trainer(max_epochs=max_epochs, use_ipex=True)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer, self.scheduler_dict)
        trainer.fit(pl_model, self.train_loader)

    def test_trainer_ipex_bf16(self):
        # IPEX BF16 weight prepack needs the cpu support avx512bw, avx512vl and avx512dq
        trainer = Trainer(max_epochs=max_epochs, use_ipex=True, precision="bf16",
                          callbacks=[CheckIPEXFusedStepCallback()])

        # use_ipex=True will perform inplace optimization
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = nn.CrossEntropyLoss()
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=max_epochs,
                steps_per_epoch=len(self.train_loader),
            ),
            "interval": "step",
        }

        pl_model = Trainer.compile(model, loss, optimizer, scheduler_dict)
        trainer.fit(pl_model, self.train_loader)
        trainer.test(pl_model, self.train_loader)

        if trainer.use_ipex and TORCH_VERSION_LESS_1_10:
            import intel_pytorch_extension as ipex
            # Diable IPEX AMP
            # Avoid affecting other tests
            ipex.enable_auto_mixed_precision(None)

    def test_trainer_ipex_bf16_unspport_optim(self):
        # IPEX BF16 weight prepack needs the cpu support avx512bw, avx512vl and avx512dq
        trainer = Trainer(max_epochs=max_epochs, use_ipex=True, precision="bf16",
                          callbacks=[CheckIPEXFusedStepCallback()])

        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss = nn.CrossEntropyLoss()
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=max_epochs,
                steps_per_epoch=len(self.train_loader),
            ),
            "interval": "step",
        }

        pl_model = Trainer.compile(model, loss, optimizer, scheduler_dict)
        trainer.fit(pl_model, self.train_loader)
        trainer.test(pl_model, self.train_loader)

        if trainer.use_ipex and TORCH_VERSION_LESS_1_10:
            import intel_pytorch_extension as ipex
            # Diable IPEX AMP
            # Avoid affecting other tests
            ipex.enable_auto_mixed_precision(None)


if __name__ == '__main__':
    pytest.main([__file__])

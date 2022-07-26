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

from multiprocessing import reduction
import os
from unittest import TestCase

import pytest
import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.algorithms.selective_backprop import SelectiveBackprop

num_classes = 10
batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class CheckBatchSize(Callback):

    def __init__(self,
                 start: float = 0.5,
                 keep: float = 0.5,
                 end: float = 0.9,
                 interrupt: int = 2,
                 batch_size: int = 256):
        self.start = start
        self.keep = keep
        self.end = end
        self.interrupt = interrupt
        self.batch_size = batch_size

    def __should_selective_backprop(
        self,
        current_duration: float,
        batch_idx: int,
        start: float = 0.5,
        end: float = 0.9,
        interrupt: int = 2,
    ) -> bool:
        is_interval = ((current_duration >= start) and (current_duration < end))
        is_step = ((interrupt == 0) or ((batch_idx + 1) % interrupt != 0))

        return is_interval and is_step

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                           dataloader_idx: int) -> None:
        elapsed_duration = float(trainer.current_epoch) / \
            float(trainer.max_epochs)
        if self.__should_selective_backprop(elapsed_duration, batch_idx, self.start, self.end,
                                            self.interrupt):
            current_batch_size = len(batch[1])
            ideal_batch_size = int(self.keep * self.batch_size)
            assert current_batch_size == ideal_batch_size, \
                'Batch size is not right.'


class ResNet18(nn.Module):

    def __init__(self, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = torch.nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


model = ResNet18(pretrained=False, include_top=False, freeze=True)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


class TestLightningModuleFromTorch(TestCase):

    def test_selective_backprop(self):
        pl_model = LightningModule(
            model,
            loss,
            optimizer,
            metrics=[torchmetrics.F1(num_classes),
                     torchmetrics.Accuracy(num_classes=10)])
        data_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        sb = SelectiveBackprop(start=0.25,
                               keep=0.53,
                               end=0.5,
                               scale_factor=0.1,
                               interrupt=2,
                               loss_fn=loss_fn)
        batch_size_check = CheckBatchSize(start=0.25,
                                          keep=0.53,
                                          end=0.5,
                                          interrupt=2,
                                          batch_size=batch_size)
        trainer = Trainer(max_epochs=4,
                          log_every_n_steps=1,
                          algorithms=[sb],
                          callbacks=[batch_size_check])
        trainer.fit(pl_model, data_loader, data_loader)


if __name__ == '__main__':
    test = TestLightningModuleFromTorch()
    test.test_selective_backprop()
    pytest.main([__file__])

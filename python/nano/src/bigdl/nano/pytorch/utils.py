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

import operator
from pytorch_lightning.utilities.imports import _compare_version
from types import MethodType
import pytorch_lightning as pl
from typing import Optional
import torch

TORCH_VERSION_LESS_1_10 = _compare_version("torch", operator.lt, "1.10")
TORCH_VERSION_LESS_1_11 = _compare_version("torch", operator.lt, "1.11")
TORCH_VERSION_LESS_1_12 = _compare_version("torch", operator.lt, "1.12")

LIGHTNING_VERSION_LESS_1_6 = _compare_version("pytorch_lightning", operator.lt, "1.6")


def wrap_data_fuction(model: pl.LightningModule):
    if not getattr(model, "on_before_batch_transfer_wrapped", None):
        fn = getattr(model, "on_before_batch_transfer")

        def on_before_batch_transfer(self, batch, dataloader_idx):
            if isinstance(batch, torch.Tensor) and batch.dim() == 4:
                batch = fn(batch, dataloader_idx)
                batch = batch.to(memory_format=torch.channels_last)
            elif isinstance(batch, list) or isinstance(batch, tuple):
                batch = list(batch)
                for index, t in enumerate(batch):
                    batch[index] = on_before_batch_transfer(self, t, dataloader_idx)
            return batch

        setattr(model, "on_before_batch_transfer_wrapped", fn)
        model.on_before_batch_transfer = MethodType(on_before_batch_transfer, model)
    else:
        setattr(model, "on_before_batch_transfer", model.on_before_batch_transfer_wrapped)
        delattr(model, "on_before_batch_transfer_wrapped")


class ChannelsLastCallback(pl.Callback):
    def setup(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        wrap_data_fuction(pl_module)
        trainer.model = trainer.model.to(memory_format=torch.channels_last)
        return super().setup(trainer, pl_module, stage)

    def teardown(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        wrap_data_fuction(pl_module)
        return super().teardown(trainer, pl_module, stage)

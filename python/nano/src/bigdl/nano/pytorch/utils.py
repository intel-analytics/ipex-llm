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


def batch_call(func):
    """
    Decorator to extending hook of pl_module.

    Extending behavior hook on_before_batch_transfer to convert data to channels_last
    at each step.
    """

    def on_before_batch_transfer(self, batch, dataloader_idx):

        def convert_channels_last(batch):
            if isinstance(batch, torch.Tensor) and batch.dim() == 4:
                batch = batch.to(memory_format=torch.channels_last)
            elif isinstance(batch, list) or isinstance(batch, tuple):
                batch = list(batch)
                for index, t in enumerate(batch):
                    batch[index] = convert_channels_last(t)
            return batch
        batch = func(batch, dataloader_idx)
        batch = convert_channels_last(batch)
        return batch
    return on_before_batch_transfer


class ChannelsLastCallback(pl.Callback):
    """Custom pl.Callback for converting model and data to channels_last."""

    def setup(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        """Override hook setup to convert model to channels_last and wrap DataHook."""
        # TODO: Add check for module_states
        fn_old = getattr(pl_module, "on_before_batch_transfer")
        fn = batch_call(fn_old)
        setattr(pl_module, "on_before_batch_transfer_origin", fn_old)
        pl_module.on_before_batch_transfer = MethodType(fn, pl_module)
        trainer.model = trainer.model.to(memory_format=torch.channels_last)
        return super().setup(trainer, pl_module, stage)

    def teardown(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        """Undo the changes to pl_module at end of fit, validate, tests, or predict."""
        setattr(pl_module, "on_before_batch_transfer", pl_module.on_before_batch_transfer_origin)
        delattr(pl_module, "on_before_batch_transfer_origin")
        return super().teardown(trainer, pl_module, stage)

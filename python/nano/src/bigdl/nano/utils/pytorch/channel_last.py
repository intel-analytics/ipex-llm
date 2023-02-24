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

from typing import Optional
from types import MethodType
from logging import warning
import torch
import pytorch_lightning as pl


def generate_channels_last_available(inputs):
    '''
    This function will generate a list of string to decide if the
    elements of input can be converted to

    channel_last: "channel_last"
    channel_last_3d: "channel_last_3d"
    no change: "original"
    '''
    # try channels_last available
    if inputs is not None:  # to avoid the situation of inputs == None
        if isinstance(inputs, torch.Tensor):
            inputs = tuple([inputs])  # make it an tuple for later process
        channels_last_available = ["original"] * len(inputs)
        for idx, input in enumerate(inputs):
            try:
                input.to(memory_format=torch.channels_last)
                channels_last_available[idx] = "channels_last"
            except Exception as _e:
                try:
                    input.to(memory_format=torch.channels_last_3d)
                    channels_last_available[idx] = "channels_last_3d"
                except Exception as _e:
                    pass
    else:
        channels_last_available = []
    return channels_last_available


def apply_proper_channels_last(flag, input_item):
    '''
    This function will apply proper channes_last to
    input item. flag has 3 possible values:

    channel_last: "channel_last"
    channel_last_3d: "channel_last_3d"
    no change: "original"
    '''
    if flag == "channels_last":
        return input_item.to(memory_format=torch.channels_last)
    if flag == "channels_last_3d":
        return input_item.to(memory_format=torch.channels_last_3d)
    return input_item


def batch_call(func):
    """
    Decorator to extending hook of pl_module.

    Extending behavior hook on_before_batch_transfer to convert data to channels_last
    for each batch.
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
        try:
            pl_module = pl_module.to(memory_format=torch.channels_last)
        except Exception as e:
            warning(f"Convert model to channels last failed, \
                    fall back to origin memory format. Exception msg: {e}")
            return super().setup(trainer, pl_module, stage)
        fn_old = getattr(pl_module, "on_before_batch_transfer")
        fn = batch_call(fn_old)
        setattr(pl_module, "on_before_batch_transfer_origin", fn_old)
        pl_module.on_before_batch_transfer = MethodType(fn, pl_module)
        return super().setup(trainer, pl_module, stage)

    def teardown(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        """Undo the changes to pl_module at end of fit, validate, tests, or predict."""
        if hasattr(pl_module, "on_before_batch_transfer_origin"):
            setattr(pl_module, "on_before_batch_transfer",
                    pl_module.on_before_batch_transfer_origin)
            delattr(pl_module, "on_before_batch_transfer_origin")
        return super().teardown(trainer, pl_module, stage)

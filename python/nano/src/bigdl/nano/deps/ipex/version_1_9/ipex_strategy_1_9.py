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

from typing import Union, Dict, Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import PrecisionPlugin, MixedPrecisionPlugin
from bigdl.nano.utils.log4Error import invalidInputError
import intel_pytorch_extension as ipex
from pytorch_lightning.strategies import SingleDeviceStrategy

from .ipex_torchfunctional import to_cpu
from .ipex_accelerator_1_9 import IPEXAccelerator

_STEP_OUTPUT_TYPE = Union[torch.Tensor, Dict[str, Any]]

class IPEXStrategy(SingleDeviceStrategy):

    def __init__(
        self,
        accelerator: Accelerator = IPEXAccelerator(),
        precision_plugin: PrecisionPlugin = PrecisionPlugin(),
        enable_bf16=False,
    ) -> None:
        """
        Args:
            accelerator: the accelerator to handle hardware
            precision_plugin: the plugin to handle precision-specific parts
        """        
        if enable_bf16:
            # Automatically mix precision
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)

        self.device = ipex.DEVICE

        super().__init__(accelerator=accelerator, precision_plugin=precision_plugin)

    def setup(self, trainer: pl.Trainer) -> None:
        """
        Setup plugins for the trainer fit and creates optimizers.
        Args:
            trainer: the trainer instance
            kwargs['model']: pl.LightningModule
        throw:
            RuntimeError:
                If AMP is used with XPU.
        """

        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            invalidInputError(False, "amp is not supported in bigdl-nano.")

        super().setup(trainer)
        
    def training_step_end(self, output: _STEP_OUTPUT_TYPE) -> _STEP_OUTPUT_TYPE:
        """
        For ipex xpu tensor do not support `tensor.storage()` right now,
        which is a required operation by pytorch_lightning,
        so just move output to cpu to store it, and move it back when doing backward.
        """
        output = to_cpu(output)

        return super().training_step_end(output)

    def test_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """A hook to do something at the end of the test step
        Args:
            output: the output of the test step
        """
        output = to_cpu(output)

        return super().test_step_end(output)

    def validation_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """A hook to do something at the end of the validation step
        Args:
            output: the output of the validation step
        """
        output = to_cpu(output)

        return super().validation_step_end(output)

    def backward(self,  # type: ignore
                 closure_loss: torch.Tensor,
                 *args,
                 **kwargs) -> torch.Tensor:
        """
        Moving back loss to xpu device
        """
        closure_loss = closure_loss.to(self.device)
        return super().backward(
            closure_loss,
            *args,
            **kwargs,
        )
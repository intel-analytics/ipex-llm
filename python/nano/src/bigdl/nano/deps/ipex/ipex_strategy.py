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

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.plugins.precision import PrecisionPlugin
import intel_extension_for_pytorch as ipex


class IPEX_Strategy(Strategy):
    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        enable_bf16=False,
    ) -> None:
        """
        Args:
            precision_plugin: the plugin to handle precision-specific parts
            accelerator: the accelerator to handle hardware
        """
        self.enable_bf16 = enable_bf16
        if enable_bf16:
            # Automatically mix precision
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)

        if precision_plugin is None:
            super().__init__(precision_plugin=PrecisionPlugin())
        else:
            super().__init__(accelerator=accelerator, precision_plugin=precision_plugin)

    def setup(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        """
        Setup plugins for the trainer fit and creates optimizers.
        Args:
            trainer: the trainer instance
            model: the LightningModule
        """
        self.setup_precision_plugin()
        self.setup_optimizers(trainer)

        if len(self.optimizers) > 1:
            raise RuntimeError("Ipex does not support more than one optimizers.")
        dtype = torch.bfloat16 if self.enable_bf16 else None
        model, optimizer = ipex.optimize(model, optimizer=self.optimizers[0],
                                         inplace=True, dtype=dtype)
        self.optimizers = [optimizer]

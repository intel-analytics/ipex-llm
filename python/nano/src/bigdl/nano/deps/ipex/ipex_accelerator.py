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


from typing import Union, Dict, Any
import intel_extension_for_pytorch as ipex

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.training_type import TrainingTypePlugin
from pytorch_lightning.plugins.precision import PrecisionPlugin
from bigdl.nano.utils.log4Error import invalidInputError


_STEP_OUTPUT_TYPE = Union[torch.Tensor, Dict[str, Any]]


class IPEXAccelerator(Accelerator):

    def __init__(
        self,
        training_type_plugin: TrainingTypePlugin = SingleDevicePlugin(
            torch.device('cpu')),
        enable_bf16=False,
    ) -> None:
        """

        Args:
            precision_plugin: the plugin to handle precision-specific parts
            training_type_plugin: the plugin to handle different training routines
        """
        self.enable_bf16 = enable_bf16
        super().__init__(precision_plugin=PrecisionPlugin(),
                         training_type_plugin=training_type_plugin)

    def setup(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        """
        Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
            model: the LightningModule
        """
        self.setup_training_type_plugin(model)
        if not self.training_type_plugin.setup_optimizers_in_pre_dispatch:
            self.setup_optimizers(trainer)
        self.setup_precision_plugin()

        if len(self.optimizers) > 1:
            invalidInputError("IPEX does not support more than one optimizers.")
        dtype = torch.bfloat16 if self.enable_bf16 else None
        model, optimizer = ipex.optimize(model, optimizer=self.optimizers[0],
                                         inplace=True, dtype=dtype)
        self.optimizers = [optimizer]

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
# Some parts of this file is adapted from
# https://github.com/huggingface/peft/blob/v0.5.0/src/peft/tuners/lora.py
# 
# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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


import torch

import transformers
from transformers.utils import cached_property
from bigdl.llm.transformers.low_bit_linear import LowBitLinear
from peft.tuners.lora import LoraLayer

class TrainingArguments(transformers.TrainingArguments):

    @cached_property
    def _setup_devices(self) -> "torch.device":
        device = super()._setup_devices
        from accelerate.utils import is_xpu_available
        if is_xpu_available():
            device = torch.device("xpu")
        return device

class LoraLowBitLinear(LowBitLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        LowBitLinear.__init__(
            self,
            in_features,
            out_features,
            qtype=kwargs.get("qtype"),
            bias=kwargs.get("bias", True),
        )
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter] > 0:
            result = result.clone()
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    ).to(expected_dtype)
                    * self.scaling[self.active_adapter]
                )
            else:
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )
            result += output
        return result


@staticmethod
def _create_new_module(lora_config, adapter_name, target, **kwargs):

    bias = kwargs.pop("bias", False)

    if isinstance(target, LowBitLinear):
        low_bit_kwargs = kwargs.copy()
        low_bit_kwargs.update(
            {
                "qtype": target.qtype,
            }
        )
        new_module = LoraLowBitLinear(adapter_name,
                                      target.in_features,
                                      target.out_features,
                                      bias=bias,
                                      **low_bit_kwargs)
    else:
        raise ValueError(
            f"Target module {target} is not supported. "
            f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
        )


    return new_module

from peft.tuners.lora import LoraModel
LoraModel._create_new_module = _create_new_module
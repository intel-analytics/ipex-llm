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
# ===========================================================================
#
# This file is copied from
# https://github.com/huggingface/transformers
#
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from ipex_llm.utils.common import invalidInputError
from transformers.utils.quantization_config import QuantizationConfigMixin
from transformers.utils.quantization_config import AwqBackendPackingMethod,\
    AWQLinearVersion, QuantizationMethod


@dataclass
class AwqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can
     play with a model that has been loaded using `auto-awq` library awq quantization
     relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization.
            Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to
        `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use.
            GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 )
        backend (`AwqBackendPackingMethod`, *optional*, defaults to
        `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend.
            This is useful for users that quantize their own models using `llm-awq` library.
        modules_to_not_convert (`list`, *optional*, defaults to []):
            The modules in qblock while not quantized.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: AWQLinearVersion = AWQLinearVersion.GEMM,
        backend: AwqBackendPackingMethod = AwqBackendPackingMethod.AUTOAWQ,
        modules_to_not_convert: list = [],
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AWQ

        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version.lower()
        self.backend = backend
        self.modules_to_not_convert = modules_to_not_convert

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        invalidInputError(self.backend == AwqBackendPackingMethod.AUTOAWQ
                          or self.backend == AwqBackendPackingMethod.LLMAWQ,
                          "Only supported quantization backends in "
                          f"{AwqBackendPackingMethod.AUTOAWQ} and "
                          f"{AwqBackendPackingMethod.LLMAWQ} and "
                          f"not recognized backend {self.backend}")

        invalidInputError(self.version in [AWQLinearVersion.GEMM, AWQLinearVersion.GEMV],
                          "Only supported versions are in [AWQLinearVersion.GEMM,"
                          f"AWQLinearVersion.GEMV] - not recognized version {self.version}")

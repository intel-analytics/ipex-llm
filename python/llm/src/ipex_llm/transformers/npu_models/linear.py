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

# This file is adapted from
# https://github.com/intel/intel-npu-acceleration-library/blob/main/intel_npu_acceleration_library/nn/linear.py

#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from intel_npu_acceleration_library.quantization import quantize_tensor, compress_to_i4
from intel_npu_acceleration_library.dtypes import NPUDtype
import os
import torch
from torch.nn import Parameter
import uuid
import math
from intel_npu_acceleration_library.backend import run_matmul
from typing import Optional, Union
from ipex_llm.utils.common import invalidInputError


class Linear(torch.nn.Module):
    """Torch Linear operation NPU backend."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Initialize the Linear class.

        Args:
            weight (torch.Tensor): Linear operation weight
            bias (Optional[torch.Tensor], optional): Linear operation optional bias.
                                                     Defaults to None.
        """
        super().__init__()

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if isinstance(bias, torch.Tensor) else None
        self.outC, self.inC = self.weight.shape
        self.op_id = str(uuid.uuid4())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        if self.training:
            out = self._mm(x, self.weight, None)
        else:
            out = run_matmul(x, self.weight, None, self.op_id)

        if self.bias is None:
            return out
        return out + self.bias

    @staticmethod
    def fromTorch(
        layer: torch.nn.Linear, dtype: torch.dtype = torch.float16
    ) -> Union["Linear", "QuantizedLinear"]:
        """Generate a NPU Linear layer from a torch one.

        Args:
            layer (torch.nn.Linear): the original torch.nn.Linear model to run on the NPU
            dtype (torch.dtype): the desired datatype

        Returns:
            Union[Linear, QuantizedLinear]: A NPU linear layer
        """
        if any(dim > 2**17 for dim in layer.weight.shape):
            return layer
        return Linear.fromTensor(layer.weight, getattr(layer, "bias", None), dtype)

    @staticmethod
    def fromTensor(
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        dtype: torch.dtype = torch.float16,
    ) -> Union["Linear", "QuantizedLinear"]:
        """Generate a NPU Linear layer from a torch one.

        Args:
            weight (torch.Tensor): the original weight tensor
            bias (Optional[torch.Tensor]): the original bias tensor
            dtype (torch.dtype): the desired datatype

        Raises:
            RuntimeError: dtype not supported

        Returns:
            Union[Linear, QuantizedLinear]: A NPU linear layer
        """
        if dtype.is_floating_point:
            if bias is None:
                return Linear(weight.to(dtype), None)
            return Linear(weight.to(dtype), bias.to(dtype))
        elif isinstance(dtype, NPUDtype):
            weights_quant, scale = quantize_tensor(weight, (dtype.min, dtype.max))
            if dtype.bits == 4:
                weights_quant = compress_to_i4(weights_quant)
            return QuantizedLinear(weights_quant, scale, bias)
        elif dtype == torch.int8:
            weights_quant, scale = quantize_tensor(weight)
            return QuantizedLinear(weights_quant, scale, bias)
        else:
            invalidInputError(False,
                              f"NPU do not support yet the requeste datatype: {dtype}")


class QuantizedLinear(torch.nn.Module):
    """Torch Quantized Linear operation NPU backend."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        qtype: Optional[str] = "sym_int4_rtn",
        group_size: int = 0,
    ):
        """Initialize the QuantizedLinear class.

        Args:
            weight (torch.Tensor): Linear operation weight
            scale (torch.Tensor): Quantization scale
            zero (Optional[torch.Tensor], optional): Quantization zero for asym_int4_rtn
            bias (Optional[torch.Tensor], optional): Linear operation optional bias.
                                                     Defaults to None.
            qtype (Optional[str], optional): qtype of this Linear

        Raises:
            RuntimeError: Quantized weight must be in torch.int8 format
        """
        super().__init__()

        self.weight = Parameter(weight, requires_grad=False).contiguous()
        if self.weight.dtype not in (torch.int8, torch.uint8):
            invalidInputError(
                False,
                (
                    f"Quantized weight must be in torch.(u)int8"
                    " dtype instead of {self.weight.dtype}"
                )
            )
        self.outC, self.inC = self.weight.shape
        self.zero = None
        if group_size != 0:
            self.scale = Parameter(scale, requires_grad=False)
            if zero is not None:
                self.zero = Parameter(zero, requires_grad=False)
        else:
            if self.weight.dtype == torch.uint8:
                # Int4 we need to double the input channels because weights are compressed
                self.inC *= 2
            self.scale = Parameter(scale * math.sqrt(self.inC), requires_grad=False)
            if zero is not None:
                self.zero = Parameter(zero * math.sqrt(self.inC), requires_grad=False)
        self.bias = bias
        self.qtype = qtype
        self.op_id = str(uuid.uuid4())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Raises:
            RuntimeError: Training is not supported for QuantizedLinear layer.
                          Use `.eval()` to do inference only

        Returns:
            torch.Tensor: result
        """

        # we assume a Linear is lm_head when its out_features > 30000,
        # if out_features > 100000, enable lm_head optimization automatically
        if x.size(1) > 500 and (
            (self.outC > 100_000 and os.environ.get("IPEX_LLM_LAST_LM_HEAD") != "0") or
            (self.outC > 30_000 and os.environ.get("IPEX_LLM_LAST_LM_HEAD") == "1")
        ):
            x = x[:, -1:, :]
        if self.training:
            invalidInputError(
                False,
                (
                    "Training is not supported for QuantizedLinear layer."
                    "Use `.eval()` to do inference only"
                )
            )

        zero_data = self.zero.data if self.zero is not None else None
        out = run_matmul(x, self.weight.data, self.scale.data, zero_data, self.op_id)

        if self.bias is None:
            return out
        return out + self.bias


class DequantizedLinear(torch.nn.Module):
    """Torch Quantized Linear operation NPU backend."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        qtype: Optional[str] = "sym_int4_rtn",
    ):
        """Initialize the DequantizedLinear class.
        Args:
            weight (torch.Tensor): Linear operation quantized weight
            scale (torch.Tensor): Quantization scale
            zero (Optional[torch.Tensor], optional): Quantization zero for asym_int4_rtn
            bias (Optional[torch.Tensor], optional): Linear operation optional bias.
                                                     Defaults to None.
            qtype (Optional[str], optional): qtype of this Linear
        Raises:
            RuntimeError: Quantized weight must be in torch.int8 format
        """
        super().__init__()

        if weight.dtype not in (torch.int8, torch.uint8):
            invalidInputError(
                False,
                (
                    f"Quantized weight must be in torch.(u)int8"
                    " dtype instead of {self.weight.dtype}"
                )
            )

        if weight.dtype == torch.uint8:
            weight = weight.view(torch.int8)
            high_4bits = weight >> 4
            low_4bits = (weight << 4) >> 4

            combined_weight = torch.cat((low_4bits.unsqueeze(2), high_4bits.unsqueeze(2)), dim=2)
            decompressed_weight = combined_weight.view(combined_weight.size(0), -1)
            dequantized_weight = decompressed_weight.to(torch.float32) * \
                torch.unsqueeze(scale.to(torch.float32), dim=1)
            if qtype == "asym_int4_rtn" and zero is not None:
                dequantized_weight = dequantized_weight + torch.unsqueeze(zero.to(torch.float32),
                                                                          dim=1)
            self.weight = Parameter(dequantized_weight, requires_grad=False).contiguous()
        else:
            dequantized_weight = weight.to(torch.float32) * \
                torch.unsqueeze(scale.to(torch.float32), dim=1)
            self.weight = Parameter(dequantized_weight.to(torch.float32),
                                    requires_grad=False).contiguous()

        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.
        Args:
            x (torch.Tensor): Input tensor
        Raises:
            RuntimeError: Training is not supported for DequantizedLinear layer.
                          Use `.eval()` to do inference only
        Returns:
            torch.Tensor: result
        """

        if self.training:
            invalidInputError(
                False,
                (
                    "Training is not supported for DequantizedLinear layer."
                    "Use `.eval()` to do inference only"
                )
            )

        out = torch.matmul(x.to(torch.float32), torch.transpose(self.weight.data, 0, 1))

        if self.bias is None:
            return out
        return out + self.bias

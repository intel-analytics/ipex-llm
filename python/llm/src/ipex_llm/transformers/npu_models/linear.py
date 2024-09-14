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
from intel_npu_acceleration_library.backend import NNFactory, run_matmul
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
from typing import Optional, Dict, Deque, Union
from functools import partial
from collections import deque
import numpy as np
from ipex_llm.utils.common import invalidInputError


_model_cache: Dict[str, Deque[NNFactory]] = {}


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


class LMHeadLinear(NNFactory):
    """Quantized Linear class, computing a matrix matrix multiplication with weights prefetching."""

    def __init__(
        self,
        inC: int,
        outC: int,
        batch: int,
        split_num: int = 2,
        profile: bool = False,
        device: str = "NPU",
        dtype: np.dtype = np.int8,
    ):
        """Initialize the QLinear class.

        Args:
            inC (int): input channels
            outC (int): output channels
            batch (int): batch
            profile (bool): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
            dtype (np.dtype): weights datatype. Defaults to np.int8.

        """
        super().__init__(profile, device)
        self.inC, self.outC = inC, outC
        self.batch = batch

        input = self.parameter((self.batch, self.inC))

        self.split_num = split_num
        split_size = self.inC // split_num // 2 * 2

        for i in range(self.split_num):
            start_idx = i * split_size
            if i == split_num - 1:
                end_idx = self.inC
            else:
                end_idx = (i + 1) * split_size

            input_slice = self.slice(input, begin=[0, start_idx],
                                     end=[self.batch, end_idx])
            linear_slice = self.linear(input_slice, outC, split_size, bias=False, wt_dtype=dtype)

            if i == 0:
                res = linear_slice
            else:
                res += linear_slice

        print("start compiling lm_head")
        self.compile()
        print("end compiling lm_head")

    def run(
        self, X: np.ndarray
    ) -> np.ndarray:
        """Run the layer:  $X * (W * S)^T$ .

        Args:
            X (np.ndarray): activation

        Raises:
            RuntimeError: Input, weights or scale shape mismatch

        Returns:
            np.ndarray: result
        """
        self.prefetchWeights(1, verify_size=False)

        self.set_input_tensor(X, 0)
        self.elapsed = backend_lib.run(self._mm)
        if len(self.out) == 1:
            return self.out[0]
        return self.out


class QuantizedLinear(torch.nn.Module):
    """Torch Quantized Linear operation NPU backend."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """Initialize the QuantizedLinear class.

        Args:
            weight (torch.Tensor): Linear operation weight
            scale (torch.Tensor): Quantization scale
            bias (Optional[torch.Tensor], optional): Linear operation optional bias.
                                                     Defaults to None.

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
        if self.weight.dtype == torch.uint8:
            # In case is Int4 we need to double the input channels because weights are compressed
            self.inC *= 2
        self.scale = Parameter(scale * math.sqrt(self.inC), requires_grad=False)
        self.bias = bias
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

        out = run_matmul(x, self.weight.data, self.scale.data, self.op_id)

        if self.bias is None:
            return out
        return out + self.bias

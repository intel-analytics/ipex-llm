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
from intel_npu_acceleration_library.nn.autograd import AutogradMatMul
from intel_npu_acceleration_library.backend import run_matmul
from intel_npu_acceleration_library.dtypes import NPUDtype
import os
import torch
from torch.nn import Parameter
import uuid
import math
from intel_npu_acceleration_library.backend import Linear, QLinear
from intel_npu_acceleration_library.backend import MatMul, QMatMul
from intel_npu_acceleration_library.backend import NNFactory
from intel_npu_acceleration_library.backend.runtime import set_contiguous, adapt_output_tensor
from typing import Optional, Any, List, Dict, Deque, Union
from functools import partial
from collections import deque
import numpy as np
from ipex_llm.utils.common import invalidInputError


_model_cache: Dict[str, Deque[NNFactory]] = {}


def clear_cache():
    """Clear the cache of models."""
    global _model_cache
    _model_cache = {}


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
        self._mm = AutogradMatMul.apply

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


class QMatMul(NNFactory):
    """Quantized Linear class, computing a matrix matrix multiplication."""

    def __init__(
        self,
        inC: int,
        outC: int,
        batch: int,
        op_id: Optional[str] = None,
        profile: bool = False,
        device: str = "NPU",
        dtype: np.dtype = np.int8,
    ):
        """Initialize the QMatmul class.

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
        self.op_id = op_id
        input = self.parameter((self.batch, self.inC))
        _ = self.linear(input, outC, inC, bias=False, wt_dtype=dtype)
        self.compile()

    def run(self, X: np.ndarray, W: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Run the layer:  X * (W * S)^T.

        Args:
            X (np.ndarray): activation
            W (np.ndarray): quantized weights
            scale (np.ndarray): quantization scale

        Raises:
            RuntimeError: Input, weights or scale shape mismatch

        Returns:
            np.ndarray: result
        """
        if not (X.shape[0] == self.batch and X.shape[1] == self.inC):
            raise RuntimeError(
                f"Input shape {X.shape} different from expected one {(self.batch, self.inC)}"
            )

        return super().run(X, (W, scale), {"op_id": self.op_id})


@torch.no_grad()
def run_matmul(
    x: torch.Tensor,
    inC: int,
    outC: int,
    weights: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    op_id: Optional[str] = None,
) -> torch.Tensor:
    """Run a matmul operation. Depending on the datatype of the weights it runs a float or quantized operation.

    Args:
        x (torch.Tensor): Activation tensor. Its dtype must be torch.float16
        weights (torch.Tensor): Weights tensor.  Its dtype can be torch.float16 or torch.int8
        scale (Optional[torch.Tensor], optional): Quantization scale. If weights.dtype == torch.int8 then it must be set. Defaults to None.
        op_id (Optional[str], optional): Operation ID. Defaults to None.

    Raises:
        RuntimeError: Unsupported weights datatype. Supported types: [torch.float16, torch.int8]

    Returns:
        torch.Tensor: result
    """
    global _model_cache

    # Set tensors as contiguous in memory
    x = set_contiguous(x)

    if weights.dtype in (torch.int8, torch.uint8):
        op_class = QMatMul
        op_class_name = op_class.__name__
        np_dtype = np.int8 if weights.dtype == torch.int8 else np.uint8
        create_op = partial(op_class, dtype=np_dtype)
        op_args = [weights.numpy(), scale.numpy()]
    else:
        raise RuntimeError(f"Unsupported dtype for weights {weights.dtype}")

    if not x.dtype.is_floating_point:
        raise RuntimeError(f"Unsupported dtype for activation {x.dtype}")

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    original_input_shape = x.shape
    expected_output_shape = list(original_input_shape[:-1]) + [outC]

    # Reshape input
    input_dtype = x.dtype
    x = x.to(torch.float16) if input_dtype != torch.float16 else x
    if len(x.shape) > 2 or x.shape[-1] != inC:
        x = x.view([-1, inC])
    x_np = x.numpy()
    batch = x_np.shape[0]

    key = f"{str(op_class_name)}_{batch}_{inC}_x_{outC}_{inC}_{x_np.dtype}"
    models = _model_cache.get(key, None)

    if models is None:
        _model_cache[key] = deque([create_op(inC, outC, batch)])
    elif len(models) < 1:
        _model_cache[key].append(create_op(inC, outC, batch))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    ret = model.run(x_np, *op_args, **op_kwargs)

    return adapt_output_tensor(ret, expected_output_shape, input_dtype)


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
        out = run_matmul(x, self.inC, self.outC, self.weight.data, self.scale.data, self.op_id)

        if self.bias is None:
            return out
        return out + self.bias

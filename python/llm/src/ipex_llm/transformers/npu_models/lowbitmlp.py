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

from ipex_llm.transformers.npu_models.runtime import run_model
from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import Optional, Sequence, List
from functools import partial
import numpy as np
import torch
import uuid


class LowBitMLP(NNFactory):
    """Computing a LowBit MLP with weights prefetching."""

    def __init__(
        self,
        input_shape: Sequence[int],
        intermediate_size: int,
        activation: str = "swiglu",
        bias: Optional[bool] = False,
        dtype: np.dtype = np.int8,
        profile: bool = False,
        device: str = "NPU",
        **additional_args
    ):
        """Initialize the LowBitMLP class.

        Args:
            input_shape (Sequence[int]): input shape channels.
            intermediate_size (int): intermediate_size of the MLP.
            activation (str): activation function to use.
            bias (Optional[bool]): Enable/Disable bias. Defaults to False.
            dtype (np.dtype): parameter type np.int8, np.uint8 and np.float16 supported.
                              Defaults to np.int8. Unit8 represents packed i4 dtypes.
            profile (bool): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
            additional_args: additional arguments
        """
        super().__init__(profile, device)
        self.intermediate_size = intermediate_size
        self.batch, self.hidden_size = input_shape
        input = self.parameter((self.batch, self.hidden_size))

        mm1 = self.linear(input, self.intermediate_size, self.hidden_size,
                          bias=bias, wt_dtype=dtype)

        if activation == "swiglu":
            mm2 = self.linear(input, self.intermediate_size, self.hidden_size,
                              bias=bias, wt_dtype=dtype)  # type: ignore[attr-defined]
            mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]
        elif activation == "clamp":
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1, additional_args.get("min"), additional_args.get("max"))
        elif activation == "elu":
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1, additional_args.get("alpha", 1.0))
        elif activation == "grn":
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1, additional_args.get("grn_bias"))
        else:
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1)

        _ = self.linear(mm1, self.hidden_size, self.intermediate_size, bias=bias, wt_dtype=dtype)
        self.compile()


class FusedLlamaLowBitMLP(torch.nn.Module):
    """LLAMA LowBit MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
    ):
        """Initialize LLAMA LowBit MLP operation.

        Args:
            parameters (List[torch.Tensor]): model weights
        """
        super().__init__()
        self.op_parameters = parameters
        self.op_id = str(uuid.uuid4())
        if isinstance(parameters[0], tuple):  # weight, scale from QuantizedLinear
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
            intermediate_size, _ = parameters[0][0].shape
        else:  # FP16 Linear
            np_dtype = np.float16
            intermediate_size, _ = parameters[0].shape
        self.backend_cls = partial(LowBitMLP, intermediate_size=intermediate_size, dtype=np_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        # Handle 3D input shape (similarly done in run_matmul)
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view([-1, x.shape[-1]])
        output = run_model(x, self.op_parameters, self.backend_cls, self.op_id)
        return output.view(original_shape)

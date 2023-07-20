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
# https://github.com/TimDettmers/bitsandbytes/blob/0.39.1/bitsandbytes/nn/modules.py
# which is licensed under the MIT license:
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional, TypeVar, Union, overload
from bigdl.llm.utils.common import invalidInputError

import torch
import torch.nn.functional as F
from torch import Tensor, device, dtype, nn

T = TypeVar("T", bound="torch.nn.Module")

import bigdl.llm.ggml.model.llama.llama_cpp as ggml
from bigdl.llm.utils.isa_checker import is_server

import torch
import ctypes
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
IS_SERVER = is_server()
TORCH_LINEAR_THRESHOLD = 96
SYM_INT4 = ggml_tensor_qtype["sym_int4"]


def ggml_convert_quant(tensor: torch.Tensor, qtype: int, convert_shape_only=False):
    QK = ggml.ggml_qk_size(qtype)
    block_size_in_bytes = ggml.ggml_type_size(qtype)

    invalidInputError(tensor.dtype == torch.float,
                      "Input tensor must be float32")
    src = tensor.data.data_ptr()
    src = ctypes.cast(src, ctypes.POINTER(ctypes.c_float))
    n = tensor.numel()
    invalidInputError(n % QK == 0,
                      "Input tensor size must be multiple of 64")
    k = tensor.shape[-1]
    invalidInputError(k % QK == 0,
                      "Last dim of input tensor must be multiple of 64")

    dst_size = (n // QK) * block_size_in_bytes
    dst_tensor = torch.empty(dst_size, dtype=torch.uint8)
    dst = ctypes.c_void_p(dst_tensor.data.data_ptr())

    hist = (ctypes.c_int64 * 16)()

    if not convert_shape_only:
        ggml.ggml_quantize_tensor(src, dst, qtype, n, k, hist)
    return dst_tensor


def ggml_int4_convert_fp32(tensor: torch.Tensor, weight_shape: tuple, k: int):
    invalidInputError(tensor.dtype == torch.uint8,
                      "Input tensor must be uint8")
    src_ptr = ctypes.c_void_p(tensor.data.data_ptr())

    dst_size = k
    dst_tensor = torch.empty(weight_shape, dtype=torch.float)
    dst_ptr = ctypes.c_void_p(dst_tensor.data.data_ptr())

    ggml.ggml_dequantize_q4_0(src_ptr, dst_ptr, k)
    return dst_tensor


class ParamsQuant(torch.nn.Parameter):
    def __new__(cls,
                data=None,
                requires_grad=True,
                old_data=None,
                quantized=False,
                _shape=None,
                convert_shape_only=False,
                qtype=None):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.data = data
        self.quantized = quantized
        self._shape = _shape
        self.convert_shape_only = convert_shape_only
        self.qtype = qtype
        return self

    def quantize(self, device):
        if not self.quantized:
            w = self.data.contiguous().float()
            # self.old_data = self.data
            w_quantized = ggml_convert_quant(w, self.qtype,
                                             convert_shape_only=self.convert_shape_only)
            self.data = w_quantized
            self.quantized = True
            self._shape = w.shape
        return self

    def get_shape(self):
        return self._shape

    @overload
    def to(self: T, device: Optional[Union[int, device]]=...,
           dtype: Optional[Union[dtype, str]]=..., non_blocking: bool=...,) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool=...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool=...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if (device is not None and device.type == "cpu" and self.data.device.type == "cpu"):
            return self.quantize(device)
        else:
            new_param = ParamsQuant(super().to(device=device,
                                               dtype=dtype,
                                               non_blocking=non_blocking),
                                    requires_grad=self.requires_grad,
                                    quantized=self.quantized,
                                    _shape=self._shape,
                                    qtype=self.qtype)

            return new_param


def ggml_matmul_src1_x_src0_t(src0: torch.Tensor,
                              src1: torch.Tensor,
                              src0_shape: torch.Size,
                              src0_qtype: int):
    if src1.dtype != torch.float32:
        src1 = src1.float()

    src0_ptr = src0.data_ptr()
    src1_ptr = src1.data_ptr()

    result_shape = (src1.shape[0], src0_shape[0])

    result_t = torch.empty(result_shape, dtype=torch.float32)
    result_ptr = result_t.data_ptr()

    src0_shape = tuple(reversed(src0_shape))
    src1_shape = tuple(reversed(src1.shape))

    # ctx_p = ctx.context
    src_0_ne = (ctypes.c_int64 * 2)(*src0_shape)
    src_0_data = ctypes.c_void_p(src0_ptr)
    src_1_ne = (ctypes.c_int64 * 2)(*src1_shape)
    src_1_data = ctypes.c_void_p(src1_ptr)
    result_ptr = ctypes.c_void_p(result_ptr)

    ggml.ggml_compute_forward_mul_mat_q_fp32(
        # ctx=ctx_p,
        src_0_ne=src_0_ne,
        src_0_data=src_0_data,
        src_0_qtype=src0_qtype,
        src_1_ne=src_1_ne,
        src_1_data=src_1_data,
        result=result_ptr,
    )

    return result_t


class LinearQuant(nn.Linear):
    def __init__(self, input_features, output_features, qtype, bias=True):
        super().__init__(input_features, output_features, bias)
        self.weight = ParamsQuant(self.weight.data, requires_grad=False,
                                  old_data=self.weight.data,
                                  quantized=False, _shape=None, qtype=qtype)
        self.in_len = input_features
        self.out_len = output_features
        self.weight_shape = (self.out_len, self.in_len)
        self.weight_length = self.out_len * self.in_len
        self.qtype = qtype

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])

        x0 = self.weight.data

        # todo may need to set a different number on different platforms
        if IS_SERVER and self.qtype == SYM_INT4 and x_2d.shape[0] >= TORCH_LINEAR_THRESHOLD:
            x0_fp32 = ggml_int4_convert_fp32(x0, self.weight_shape, self.weight_length)
            result = F.linear(x, x0_fp32, self.bias)
        else:
            result = ggml_matmul_src1_x_src0_t(x0, x_2d, self.weight_shape, self.qtype)
            new_shape = x_shape[:-1] + (self.out_len,)
            result = result.view(new_shape)
            if self.bias is not None:
                result += self.bias

        return result.to(x.dtype)

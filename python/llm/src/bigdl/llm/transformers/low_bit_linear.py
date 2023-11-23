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
import os
import torch
import torch.nn.functional as F
from torch import Tensor, device, dtype, nn
from operator import mul
from functools import reduce
from bigdl.llm.transformers.xpu_customize_fwd import custom_fwd, custom_bwd

T = TypeVar("T", bound="torch.nn.Module")

import bigdl.llm.ggml.model.llama.llama_cpp as ggml
from bigdl.llm.utils.isa_checker import is_server, is_spr
import ctypes
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
IS_SERVER = is_server()
IS_SPR = is_spr()
TORCH_LINEAR_THRESHOLD = 96
SYM_INT4 = ggml_tensor_qtype["sym_int4"]
ASYM_INT4 = ggml_tensor_qtype["asym_int4"]
SYM_INT8 = ggml_tensor_qtype["sym_int8"]
NF4 = ggml_tensor_qtype["nf4"]
NF3 = ggml_tensor_qtype["nf3"]
FP8 = ggml_tensor_qtype["fp8"]
FP4 = ggml_tensor_qtype["fp4"]
MOFQ4 = ggml_tensor_qtype["mixed_fp4"]
MOFQ8 = ggml_tensor_qtype["mixed_fp8"]


def get_ggml_qk_size(qtype: str):
    return ggml.ggml_qk_size(ggml_tensor_qtype[qtype])


def ggml_convert_qtype(tensor: torch.Tensor, qtype: int,
                       device=None, convert_shape_only=False):
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
    dst_tensor = torch.empty(dst_size, dtype=torch.uint8,
                             device=device)

    if not convert_shape_only and device != 'meta':
        dst = ctypes.c_void_p(dst_tensor.data.data_ptr())
        hist = (ctypes.c_int64 * 16)()
        ggml.ggml_quantize_tensor(src, dst, qtype, n, k, hist)
    return dst_tensor


def ggml_q_format_convet_cpu2xpu(tensor: torch.Tensor, num_elem: int, qtype: int):

    invalidInputError(tensor.dtype == torch.uint8,
                      "Input tensor must be uint8")

    invalidInputError(tensor.device == torch.device('cpu'),
                      "Input tensor must be uint8")

    src = ctypes.c_void_p(tensor.data.data_ptr())

    if qtype in [SYM_INT4, ASYM_INT4, SYM_INT8, NF4, NF3, FP4, FP8]:
        dst_tensor = torch.empty_like(tensor)
    elif qtype == ggml_tensor_qtype["sym_int5"]:
        QK = ggml.ggml_qk_size(qtype)
        block_size_in_bytes = ggml.ggml_type_size(ggml_tensor_qtype["asym_int5"])
        dst_size = (num_elem // QK) * block_size_in_bytes
        dst_tensor = torch.empty(dst_size, dtype=torch.uint8,
                                 device=torch.device('cpu'))
    else:
        return tensor
    dst = ctypes.c_void_p(dst_tensor.data.data_ptr())
    ggml.ggml_q_format_convet_cpu2xpu(src, dst, num_elem, qtype)
    return dst_tensor


def ggml_q_format_convet_xpu2cpu(tensor: torch.Tensor, num_elem: int, qtype: int):

    invalidInputError(tensor.dtype == torch.uint8,
                      "Input tensor must be uint8")

    invalidInputError(tensor.device == torch.device('cpu'),
                      "Input tensor must be uint8")

    src = ctypes.c_void_p(tensor.data.data_ptr())

    if qtype in [SYM_INT4, ASYM_INT4, SYM_INT8, NF4, NF3, FP4, FP8]:
        dst_tensor = torch.empty_like(tensor)
    elif qtype == ggml_tensor_qtype["sym_int5"]:
        QK = ggml.ggml_qk_size(ggml_tensor_qtype["asym_int5"])
        block_size_in_bytes = ggml.ggml_type_size(qtype)
        dst_size = (num_elem // QK) * block_size_in_bytes
        dst_tensor = torch.empty(dst_size, dtype=torch.uint8,
                                 device=torch.device('cpu'))
    else:
        return tensor
    dst = ctypes.c_void_p(dst_tensor.data.data_ptr())
    ggml.ggml_q_format_convet_xpu2cpu(src, dst, num_elem, qtype)
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


def ggml_convert_fp32(tensor: torch.Tensor, weight_shape: tuple, k: int, qtype: int):
    invalidInputError(tensor.dtype == torch.uint8,
                      "Input tensor must be uint8")
    src_ptr = ctypes.c_void_p(tensor.data.data_ptr())

    dst_size = k
    dst_tensor = torch.empty(weight_shape, dtype=torch.float)
    dst_ptr = ctypes.c_void_p(dst_tensor.data.data_ptr())

    ggml.ggml_dequantize(src_ptr, dst_ptr, k, qtype)
    return dst_tensor


# Rename to FP4Params to trigger initializing
# the params layer with all parameters on the CPU
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py#L333
class FP4Params(torch.nn.Parameter):
    def __new__(cls,
                data=None,
                requires_grad=False,
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
        self.qtype = qtype
        self.convert_shape_only = convert_shape_only
        return self

    def ggml_mse(self, w, ggml_qtype, device):
        from torch.nn.functional import mse_loss
        w_quant = ggml_convert_qtype(w, ggml_qtype,
                                     device=device,
                                     convert_shape_only=self.convert_shape_only)
        w_dequant = ggml_convert_fp32(w_quant, w.shape,
                                      reduce(mul, w.shape, 1), ggml_qtype)
        mse = mse_loss(w_dequant, w)
        return mse, w_quant

    def quantize(self, device=None):
        if not self.quantized:
            w = self.data.contiguous().float()
            if self.qtype == MOFQ4:
                if device == 'meta':
                    w_quantized = ggml_convert_qtype(w, SYM_INT4,
                                                     device=device,
                                                     convert_shape_only=self.convert_shape_only)
                    # TODO: should load from config, the current implementation doesn't support
                    # save/load
                    self.qtype = SYM_INT4
                else:
                    q4_0_mse, w_quant_q4_0 = self.ggml_mse(w, SYM_INT4, device=device)
                    fp4_mse, w_quant_fp4 = self.ggml_mse(w, FP4, device=device)
                    if q4_0_mse <= fp4_mse:
                        self.qtype = SYM_INT4
                        self.data = w_quant_q4_0
                    else:
                        self.qtype = FP4
                        self.data = w_quant_fp4
            elif self.qtype == MOFQ8:
                if device == 'meta':
                    w_quantized = ggml_convert_qtype(w, SYM_INT8,
                                                     device=device,
                                                     convert_shape_only=self.convert_shape_only)
                    # TODO: should load from config, the current implementation doesn't support
                    # save/load
                    self.qtype = SYM_INT8
                else:
                    q8_0_mse, w_quant_q8_0 = self.ggml_mse(w, SYM_INT8, device=device)
                    fp8_mse, w_quant_fp8 = self.ggml_mse(w, FP8, device=device)
                    if q8_0_mse <= fp8_mse:
                        self.qtype = SYM_INT8
                        self.data = w_quant_q8_0
                    else:
                        self.qtype = FP8
                        self.data = w_quant_fp8
            else:
                w_quantized = ggml_convert_qtype(w, self.qtype,
                                                 device=device,
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
            return self.quantize(device.type)
        elif device is not None and device.type == "meta" and self.data.device.type == "meta":
            return self.quantize(device.type)
        elif (device is not None and device.type == "xpu" and self.data.device.type == "cpu"):
            # enter xpu logic, compile linear_int4 extension at first time
            self.quantize(device)  # tensor is cpu now
            self.data = ggml_q_format_convet_cpu2xpu(self.data,
                                                     reduce(mul, self._shape, 1),
                                                     self.qtype)
            new_param = FP4Params(super().to(device=device,
                                             dtype=dtype,
                                             non_blocking=non_blocking),
                                  requires_grad=self.requires_grad,
                                  quantized=self.quantized,
                                  _shape=self._shape,
                                  qtype=self.qtype)
            return new_param
        elif (device is not None and device.type == "cpu" and self.data.device.type == "xpu"):
            new_param = FP4Params(super().to(device=device,
                                             dtype=dtype,
                                             non_blocking=non_blocking),
                                  requires_grad=self.requires_grad,
                                  quantized=self.quantized,
                                  _shape=self._shape,
                                  qtype=self.qtype)
            new_param.data = ggml_q_format_convet_xpu2cpu(new_param.data,
                                                          reduce(mul, new_param._shape, 1),
                                                          new_param.qtype)
            return new_param
        else:
            new_param = FP4Params(super().to(device=device,
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


class MatMulLowBit(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, A, weight, input_seq_size):
        ctx.is_empty = False
        import linear_q4_0
        result = linear_q4_0.forward_new(A, weight.data, weight.qtype, input_seq_size)
        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, weight)
        else:
            ctx.tensors = (None, None)
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        import linear_q4_0
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None
        req_gradA, _, _ = ctx.needs_input_grad
        A, weight = ctx.tensors
        grad_A, grad_weight = None, None
        if req_gradA:
            if torch.xpu.is_autocast_xpu_enabled():
                grad_output = grad_output.to(torch.xpu.get_autocast_xpu_dtype())
            dequant_weight = linear_q4_0.dequant(A, weight.data, weight.qtype)
            grad_A = torch.matmul(grad_output, dequant_weight.reshape(weight._shape))

        return grad_A, grad_weight, None


class MatMulLowBitCPU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, weight):
        ctx.is_empty = False
        x0_fp32 = ggml_int4_convert_fp32(weight.data, weight._shape,
                                         weight._shape[0] * weight._shape[1])
        result = torch.matmul(A, x0_fp32.T)
        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, weight)
        else:
            ctx.tensors = (None, None)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None
        req_gradA, _, = ctx.needs_input_grad
        A, weight = ctx.tensors
        grad_A, grad_weight = None, None
        if req_gradA:
            x0_fp32 = ggml_int4_convert_fp32(weight.data, weight._shape,
                                             weight._shape[0] * weight._shape[1])
            grad_A = torch.matmul(grad_output, x0_fp32.to(grad_output.dtype))
        return grad_A, grad_weight, None


class LowBitLinear(nn.Linear):
    def __init__(self, input_features, output_features, qtype, bias=True,
                 conver_to_half=True, mp_group=None):
        super().__init__(input_features, output_features, bias)
        self.weight = FP4Params(self.weight.data,
                                requires_grad=False,
                                quantized=False, _shape=None, qtype=qtype)
        self.in_len = input_features
        self.out_len = output_features
        self.weight_shape = (self.out_len, self.in_len)
        self.weight_length = self.out_len * self.in_len
        self.qtype = qtype
        self.conver_to_half = conver_to_half
        self.mp_group = mp_group

    def forward(self, x: torch.Tensor):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])

        x0 = self.weight.data

        if x0.device.type == "xpu":
            # GPU logic
            try:
                import intel_extension_for_pytorch
                import linear_q4_0
                from bigdl.llm.utils.xmx_checker import use_xmx
            except ModuleNotFoundError:
                invalidInputError(False,
                                  "Please `pip install bigdl_core_xe` first.")

            if x_2d.is_contiguous() is False:
                x_2d = x_2d.contiguous()

            input_seq_size = x_shape[1]
            if self.training and x_2d.requires_grad:
                result = MatMulLowBit.apply(x_2d, self.weight, input_seq_size)
            else:
                # current workaround to reduce first token latency of fp32 input
                # sometimes fp16 cause nan and training instability
                # disable the conversion when training
                if self.conver_to_half and x_2d.shape[0] > 1 and x_2d.dtype == torch.float32 and \
                        not use_xmx(x_2d, self.weight.qtype):
                    x_2d = x_2d.half()
                    result = linear_q4_0.forward_new(x_2d, self.weight.data, self.weight.qtype,
                                                     input_seq_size)
                    result = result.to(x.dtype)
                else:
                    result = linear_q4_0.forward_new(x_2d, self.weight.data, self.weight.qtype,
                                                     input_seq_size)
            new_shape = x_shape[:-1] + (self.out_len,)
            result = result.view(new_shape)
            if self.mp_group is not None:
                from deepspeed import comm as dist
                dist.inference_all_reduce(result, group=self.mp_group)
            if self.bias is not None:
                result += self.bias
        else:
            # CPU logic
            # todo may need to set a different number on different platforms
            invalidInputError(self.qtype != NF3 and self.qtype != NF4 and self.qtype != FP8
                              and self.qtype != FP4,
                              "NF3, NF4, FP4 and FP8 quantization are currently not"
                              " supported on CPU")
            if self.training and x.requires_grad:
                result = MatMulLowBitCPU.apply(x, self.weight)
            else:
                # convert if necessary, and compute a linear result
                if IS_SERVER and (not IS_SPR) and \
                        self.qtype == SYM_INT4 and x_2d.shape[0] >= TORCH_LINEAR_THRESHOLD:
                    x0_fp32 = ggml_int4_convert_fp32(x0, self.weight_shape, self.weight_length)
                    result = F.linear(x, x0_fp32)
                else:
                    # Weight does not need a convert
                    result = ggml_matmul_src1_x_src0_t(x0, x_2d, self.weight_shape, self.qtype)
                    new_shape = x_shape[:-1] + (self.out_len,)
                    result = result.view(new_shape)
            # allreduce to combine partial results and add bias if necessary
            if self.mp_group is not None:
                # deepspeed distibuted mode
                from deepspeed import comm as dist
                dist.inference_all_reduce(result, group=self.mp_group)
            if self.bias is not None:
                result += self.bias
        return result


class FP16Linear(nn.Linear):
    def __init__(self, input_features, output_features, qtype, bias=True,
                 conver_to_half=True, mp_group=None):
        super().__init__(input_features, output_features, bias)
        self.in_len = input_features
        self.out_len = output_features
        self.weight_shape = (self.out_len, self.in_len)
        self.weight_length = self.out_len * self.in_len
        self.qtype = qtype
        self.conver_to_half = conver_to_half
        self.mp_group = mp_group

    def forward(self, x: torch.Tensor):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])

        x0 = self.weight.data
        # only work for GPU
        invalidInputError(x0.device.type == "xpu",
                          "FP16 only works for GPU")
        try:
            import intel_extension_for_pytorch
            import linear_fp16_esimd
        except ModuleNotFoundError:
            invalidInputError(False,
                              "Please `pip install bigdl_core_xe` first.")

        if x_2d.is_contiguous() is False:
            x_2d = x_2d.contiguous()

        if x_2d.shape[0] > 1:
            # first token or batch size > 1, re-convert weight
            original_weight = self.weight.data.transpose(1, 2)
            original_weight = original_weight.reshape(self.out_len, self.in_len)
            result = F.linear(x_2d, original_weight.contiguous())
            del original_weight
        else:
            # rest token, use esimd optimization
            result = linear_fp16_esimd.forward(x_2d, self.weight.data)

        new_shape = x_shape[:-1] + (self.out_len,)
        result = result.view(new_shape)
        if self.mp_group is not None:
            from deepspeed import comm as dist
            dist.inference_all_reduce(result, group=self.mp_group)
        if self.bias is not None:
            result += self.bias

        return result.to(x.dtype)

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
from ipex_llm.utils.common import invalidInputError
import os
import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor, dtype, nn
from operator import mul
from functools import reduce
from ipex_llm.transformers.xpu_customize_fwd import custom_fwd, custom_bwd
from ipex_llm.transformers.utils import is_autocast_enabled, get_autocast_dtype
from ipex_llm.transformers.utils import get_xpu_device_name
from ipex_llm.transformers.convert import is_deepspeed_available, get_use_vllm

T = TypeVar("T", bound="torch.nn.Module")

import ipex_llm.ggml.model.llama.llama_cpp as ggml
import ctypes
from ipex_llm.ggml.quantize import ggml_tensor_qtype

TORCH_LINEAR_THRESHOLD = int(os.getenv("BIGDL_LLM_LINEAR_THRESHOLD", "512"))
SYM_INT4 = ggml_tensor_qtype["sym_int4"]
ASYM_INT4 = ggml_tensor_qtype["asym_int4"]
SYM_INT8 = ggml_tensor_qtype["sym_int8"]
NF4 = ggml_tensor_qtype["nf4"]
NF3 = ggml_tensor_qtype["nf3"]
FP8E4 = ggml_tensor_qtype["fp8_e4m3"]
FP4 = ggml_tensor_qtype["fp4"]
MOFQ4 = ggml_tensor_qtype["mixed_fp4"]
MOFQ8 = ggml_tensor_qtype["mixed_fp8"]
FP8E5 = ggml_tensor_qtype["fp8_e5m2"]
FP6 = ggml_tensor_qtype["fp6"]
FP16 = ggml_tensor_qtype["fp16"]
IQ2_XXS = ggml_tensor_qtype["gguf_iq2_xxs"]
IQ2_XS = ggml_tensor_qtype["gguf_iq2_xs"]
Q2_K = ggml_tensor_qtype["q2_k"]
IQ1_S = ggml_tensor_qtype["gguf_iq1_s"]
Q4_K = ggml_tensor_qtype["q4_k"]
Q6_K = ggml_tensor_qtype["q6_k"]
Q5_K = ggml_tensor_qtype["q5_k"]
FP6_K = ggml_tensor_qtype["fp6_k"]
SYM_INT4_RTN = ggml_tensor_qtype["sym_int4_rtn"]
SYM_INT8_RTN = ggml_tensor_qtype["sym_int8_rtn"]
ASYM_INT4_RTN = ggml_tensor_qtype["asym_int4_rtn"]
WOQ_INT4 = ggml_tensor_qtype["woq_int4"]
RTN_DTYPE = {
    SYM_INT4_RTN: torch.uint8,
    ASYM_INT4_RTN: torch.uint8,
    SYM_INT8_RTN: torch.int8,
}


def get_block_size(qtype: str):
    return ggml.ggml_qk_size(ggml_tensor_qtype[qtype])


def get_qk_size(qtype: int):
    return ggml.ggml_qk_size(qtype)


def ggml_convert_qtype(tensor: torch.Tensor, qtype: int,
                       device=None, convert_shape_only=False,
                       imatrix: torch.Tensor=None,
                       in_features: int=None,
                       enable_scale_search: bool=False):
    QK = ggml.ggml_qk_size(qtype)
    block_size_in_bytes = ggml.ggml_type_size(qtype)

    invalidInputError(tensor.dtype == torch.float,
                      "Input tensor must be float32")
    src = tensor.data.data_ptr()
    src = ctypes.cast(src, ctypes.POINTER(ctypes.c_float))
    n = tensor.numel()  # all elements
    k = tensor.shape[-1]
    invalidInputError(k % QK == 0,
                      f"Last dim of input tensor must be multiple of {QK}")

    dst_size = (n // QK) * block_size_in_bytes
    if qtype in [SYM_INT8_RTN, SYM_INT4_RTN, ASYM_INT4_RTN]:
        dst_tensor = torch.empty(dst_size, dtype=RTN_DTYPE[qtype],
                                 device=device)
        dst_tensor = dst_tensor.reshape(tensor.shape[0], tensor.shape[-1] // QK)
        if qtype == ASYM_INT4_RTN:
            scale = torch.empty((n // k) * 2, dtype=torch.float32,
                                device=device)
        else:
            scale = torch.empty(n // k, dtype=torch.float32,
                                device=device)
    elif qtype == NF4:
        # Deepspeed zero3 requires unified dtype,
        # thus here uses bfloat16 consistent to other layers
        # dst_size above is computed based on uint8, and for bfloat16,
        # buffer size should be half
        dst_tensor = torch.empty(dst_size // 2, dtype=torch.bfloat16,
                                 device=device)
    else:
        dst_tensor = torch.empty(dst_size, dtype=torch.uint8,
                                 device=device)

    if not convert_shape_only and device != 'meta':
        dst = ctypes.c_void_p(dst_tensor.data.data_ptr())
        hist = (ctypes.c_int64 * 16)()
        if qtype not in [IQ2_XXS, IQ2_XS, Q2_K, IQ1_S, Q4_K, Q6_K, Q5_K, FP6_K]:
            if qtype in [SYM_INT8_RTN, SYM_INT4_RTN, ASYM_INT4_RTN]:
                scale_ptr = ctypes.cast(scale.data.data_ptr(), ctypes.POINTER(ctypes.c_float))
                if imatrix is None:
                    ggml.ggml_quantize_tensor_rtn(src, dst, scale_ptr, qtype, n,
                                                  k, hist, enable_scale_search)
                else:
                    imatrix = imatrix.data.data_ptr()
                    imatrix = ctypes.cast(imatrix, ctypes.POINTER(ctypes.c_float))
                    ggml.ggml_quantize_tensor_rtn_with_weights(src, dst, scale_ptr,
                                                               qtype, n,
                                                               k, hist,
                                                               enable_scale_search,
                                                               imatrix)
                return dst_tensor, scale.type(torch.float16)
            else:
                ggml.ggml_quantize_tensor(src, dst, qtype, n, k, hist, enable_scale_search)
        else:
            if imatrix is not None:
                # quantize with importance matrix
                imatrix = imatrix.data.data_ptr()
                imatrix = ctypes.cast(imatrix, ctypes.POINTER(ctypes.c_float))
            # pass nrow and n_per_row
            ggml.ggml_quantize_tensor_with_weights(src, dst, qtype,
                                                   n // in_features, in_features,
                                                   hist, imatrix)
    if qtype in [SYM_INT8_RTN, SYM_INT4_RTN, ASYM_INT4_RTN]:
        return dst_tensor, scale.type(torch.float16)
    else:
        return dst_tensor


def ggml_q_format_convet_cpu2xpu(tensor: torch.Tensor, num_elem: int, qtype: int):
    if qtype == NF4:
        invalidInputError(tensor.dtype == torch.bfloat16,
                          "NF4 Input tensor must be bfloat16")
    else:
        invalidInputError(tensor.dtype == torch.uint8,
                          "Input tensor except NF4 must be uint8")

    invalidInputError(tensor.device == torch.device('cpu'),
                      "Input tensor must be on cpu")

    src = ctypes.c_void_p(tensor.data.data_ptr())

    if qtype in [SYM_INT4, ASYM_INT4, SYM_INT8, NF4, NF3, FP4, FP6, FP8E4, FP8E5,
                 Q4_K, Q6_K, FP6_K, WOQ_INT4]:
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
    if qtype == NF4:
        invalidInputError(tensor.dtype == torch.bfloat16,
                          "NF4 Input tensor must be bfloat16")
    else:
        invalidInputError(tensor.dtype == torch.uint8,
                          "Input tensor must be uint8")

    invalidInputError(tensor.device == torch.device('cpu'),
                      "Input tensor must be on cpu")

    src = ctypes.c_void_p(tensor.data.data_ptr())

    if qtype in [SYM_INT4, ASYM_INT4, SYM_INT8, NF4, NF3, FP4, FP6, FP8E4, FP8E5,
                 Q4_K, Q6_K, FP6_K, WOQ_INT4]:
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


def reshape_lm_head_input(x):
    if x.dim() > 3:
        x = x.reshape([-1, x.shape[-2], x.shape[-1]])
    shape = list(x.size())
    if shape[1] > 10:
        shape[1] = 1
        x = x[:, -1, :].view(shape)
    return x


def use_batch_forward(x: torch.Tensor, qtype: int, output_len: int):
    device_name = get_xpu_device_name(x.device)
    batch_size = x.shape[0]
    hard_condition = (
        x.dtype in [torch.float, torch.half]
        and x.shape[1] % 128 == 0
        and (
            (
                qtype in [SYM_INT4, ASYM_INT4, FP8E5, FP8E4, WOQ_INT4]
                and (
                    batch_size <= 48
                    or (
                        batch_size <= 64
                        and x.shape[1] % 256 == 0
                        and output_len % 64 == 0
                    )
                )
            )
            or (
                qtype in [SYM_INT8, FP4, FP6, Q4_K, Q6_K, WOQ_INT4]
                and batch_size <= 48
                and device_name in ["arc", "pvc", "mtl", "arl"]
                and x.shape[1] % 256 == 0
                and output_len % 32 == 0
            )
        )
    )
    if hard_condition:
        return (
            batch_size > 1
            or (device_name in ["arc"] and qtype in [SYM_INT8, FP4])
            or (device_name in ["arc", "mtl"] and qtype in [FP8E4])
            or (device_name in ["lnl"] and qtype in [SYM_INT4, WOQ_INT4] and x.shape[1] % 512 == 0)
            or (device_name in ["bmg"] and qtype in [SYM_INT4, WOQ_INT4, FP8E5])
        )
    return False


# Rename to FP4Params to trigger initializing
# the params layer with all parameters on the CPU
class FP4Params(torch.nn.Parameter):
    def __new__(cls,
                data=None,
                requires_grad=False,
                quantized=False,
                _shape=None,
                convert_shape_only=False,
                qtype=None,
                imatrix=None,
                in_features=None,
                enable_scale_search=False):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.data = data
        self.quantized = quantized
        self._shape = _shape
        self.qtype = qtype
        self.convert_shape_only = convert_shape_only
        self.imatrix = imatrix
        self.in_features = in_features
        self.enable_scale_search = enable_scale_search
        return self

    def ggml_mse(self, w, ggml_qtype, device):
        from torch.nn.functional import mse_loss
        w_quant = ggml_convert_qtype(w, ggml_qtype,
                                     device=device,
                                     convert_shape_only=self.convert_shape_only,
                                     enable_scale_search=self.enable_scale_search)
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
                                                     convert_shape_only=self.convert_shape_only,
                                                     enable_scale_search=self.enable_scale_search)
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
                                                     convert_shape_only=self.convert_shape_only,
                                                     enable_scale_search=self.enable_scale_search)
                    # TODO: should load from config, the current implementation doesn't support
                    # save/load
                    self.qtype = SYM_INT8
                else:
                    q8_0_mse, w_quant_q8_0 = self.ggml_mse(w, SYM_INT8, device=device)
                    fp8_mse, w_quant_fp8 = self.ggml_mse(w, FP8E4, device=device)
                    if q8_0_mse <= fp8_mse:
                        self.qtype = SYM_INT8
                        self.data = w_quant_q8_0
                    else:
                        self.qtype = FP8E4
                        self.data = w_quant_fp8
            else:
                w_quantized = ggml_convert_qtype(w, self.qtype,
                                                 device=device,
                                                 convert_shape_only=self.convert_shape_only,
                                                 imatrix=self.imatrix,
                                                 in_features=self.in_features,
                                                 enable_scale_search=self.enable_scale_search)
                self.data = w_quantized
            self.quantized = True
            self._shape = w.shape
        return self

    def get_shape(self):
        return self._shape

    @overload
    def to(self: T, device: Optional[Union[int, torch.device]]=...,
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
            self.quantize("cpu")  # tensor is cpu now
            self.data = ggml_q_format_convet_cpu2xpu(self.data,
                                                     reduce(mul, self._shape, 1),
                                                     self.qtype)
            new_param = FP4Params(super().to(device=device,
                                             dtype=dtype,
                                             non_blocking=non_blocking),
                                  requires_grad=self.requires_grad,
                                  quantized=self.quantized,
                                  _shape=self._shape,
                                  qtype=self.qtype,
                                  enable_scale_search=self.enable_scale_search)
            return new_param
        elif (device is not None and device.type == "cpu" and self.data.device.type == "xpu"):
            new_param = FP4Params(super().to(device=device,
                                             dtype=dtype,
                                             non_blocking=non_blocking),
                                  requires_grad=self.requires_grad,
                                  quantized=self.quantized,
                                  _shape=self._shape,
                                  qtype=self.qtype,
                                  enable_scale_search=self.enable_scale_search)
            ggml_xpu = new_param.data
            new_param.data = ggml_q_format_convet_xpu2cpu(ggml_xpu,
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
                                  qtype=self.qtype,
                                  enable_scale_search=self.enable_scale_search)
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
    def forward(ctx, A, weight, output_size):
        ctx.is_empty = False
        import xe_linear
        if weight.qtype == NF4:
            result = xe_linear.forward_new(A,
                                           weight.data.view(torch.uint8),
                                           weight.qtype,
                                           output_size)
        else:
            result = xe_linear.forward_new(A, weight.data, weight.qtype, output_size)
        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, weight)
        else:
            ctx.tensors = (None, None)
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        import xe_linear
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None
        req_gradA, _, _ = ctx.needs_input_grad
        A, weight = ctx.tensors
        grad_A, grad_weight = None, None
        if req_gradA:
            if is_autocast_enabled("xpu"):
                grad_output = grad_output.to(get_autocast_dtype("xpu"))
            if weight.qtype == NF4:
                dequant_weight = xe_linear.dequant(A,
                                                   weight.data.view(torch.uint8),
                                                   weight.qtype)
            else:
                dequant_weight = xe_linear.dequant(A, weight.data, weight.qtype)
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
                 conver_to_half=True, mp_group=None,
                 optimize_lm_head=False, act_order=False,
                 enable_scale_search=False):
        super().__init__(input_features, output_features, bias)
        self.weight = FP4Params(self.weight.data,
                                requires_grad=False,
                                quantized=False, _shape=None, qtype=qtype,
                                enable_scale_search=enable_scale_search)
        self.in_len = input_features
        self.out_len = output_features
        self.weight_shape = (self.out_len, self.in_len)
        self.weight_length = self.out_len * self.in_len
        self.qtype = qtype
        self.conver_to_half = conver_to_half
        self.mp_group = mp_group
        self.compute_dtype = None  # only for training
        self.optimize_lm_head = optimize_lm_head
        self.device = None  # detected only once in the first forward
        # empty cache before and after lm_head at first token (by default on arc)
        # especially for baichuan/qwen when given long input at inference time.
        # The condition makes sure that empty cache only takes effect if this layer is lm_head.
        # For other models like llama, lm_cache will be applied as well
        # since performance isn't impacted.
        self.is_lm_head = self.in_len * self.out_len >= 32000 * 4096 and self.bias is None
        self.low_memory_mode = self.is_lm_head
        self.act_order = act_order
        if act_order:
            self.register_buffer(
                "g_idx_map",
                torch.tensor([i for i in range(self.in_len)], dtype=torch.int64))

    def forward(self, x: torch.Tensor):
        # empty cache before and after lm_head at first token when input > 1024
        # on arc or IPEX_LLM_LOW_MEM is set to 1 at inference time.
        if self.device is None:
            self.device = get_xpu_device_name(self.weight.data.device)
            self.low_memory_mode = \
                self.low_memory_mode and \
                (self.device == "arc" or os.environ.get("IPEX_LLM_LOW_MEM", None) == "1")
        # Due to inconsistent training status in some models like Baichuan-7b-Chat,
        # we should check both self.training and torch.is_inference_mode_enabled().
        is_training = self.training and not torch.is_inference_mode_enabled()
        if is_training:
            # below logic is only for training
            autocast_dtype = get_autocast_dtype(x.device.type)
            if self.compute_dtype is not None and x.device.type == "xpu":
                x = x.to(self.compute_dtype)  # solve GC issue for unlora module
            elif autocast_dtype is not None:
                x = x.to(autocast_dtype)

        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if self.optimize_lm_head:
            x = reshape_lm_head_input(x)

        # [batch, seq_len, in_len] -> [batch, seq_len, out_len]
        new_shape = x.shape[:-1] + (self.out_len,)

        # Activation is empty tensor, e.g., [1, 0, 4096]
        if 0 in x.shape:
            # return empty tensor with output shape, x.dtype and x.device
            return torch.empty(new_shape, dtype=x.dtype, device=x.device)

        if self.act_order:
            x = x[..., self.g_idx_map]

        x_2d = x.contiguous().view(-1, x.shape[-1])

        if self.weight.device.type == "xpu":
            if is_training and x_2d.requires_grad:
                result = MatMulLowBit.apply(x_2d, self.weight, self.out_len)
            else:
                do_empty_cache = self.low_memory_mode and x_2d.shape[0] >= 1024
                if do_empty_cache:
                    torch.xpu.empty_cache()

                if self.qtype == NF4:
                    w = self.weight.data.view(torch.uint8)
                else:
                    w = self.weight.data

                if use_batch_forward(x_2d, self.weight.qtype, self.out_len):
                    import xe_batch
                    result = xe_batch.batch_forward(x_2d, w, self.qtype)
                elif not is_training and self.conver_to_half \
                        and x_2d.shape[0] > 1 and x_2d.dtype == torch.float:
                    import xe_linear
                    x_2d = x_2d.half()
                    result = xe_linear.forward_new(x_2d, w, self.qtype, self.out_len)
                    result = result.to(x.dtype)
                else:
                    import xe_linear
                    result = xe_linear.forward_new(x_2d, w, self.qtype, self.out_len)

                if do_empty_cache:
                    torch.xpu.empty_cache()

            result = result.view(new_shape)

            if self.mp_group is not None:
                if get_use_vllm():
                    result = self.mp_group.all_reduce(result)
                elif is_deepspeed_available():
                    from deepspeed import comm as dist
                    dist.inference_all_reduce(result, group=self.mp_group)
                else:
                    invalidInputError(False, "mp_group is not None, but no supported backend found")

            if self.bias is not None:
                result += self.bias
        else:
            # CPU logic
            # todo may need to set a different number on different platforms
            invalidInputError(self.qtype != NF3 and self.qtype != NF4 and self.qtype != FP8E4
                              and self.qtype != FP4 and self.qtype != FP8E5,
                              "NF3, NF4, FP4 and FP8 quantization are currently not"
                              " supported on CPU")
            if self.training and x.requires_grad:
                result = MatMulLowBitCPU.apply(x, self.weight)
            else:
                from ipex_llm.utils.isa_checker import is_server, is_spr
                x0 = self.weight.data
                # convert if necessary, and compute a linear result
                if is_server() and (not is_spr()) and \
                        self.qtype == SYM_INT4 and x_2d.shape[0] >= TORCH_LINEAR_THRESHOLD:
                    x0_fp32 = ggml_int4_convert_fp32(x0, self.weight_shape, self.weight_length)
                    result = F.linear(x.to(dtype=x0_fp32.dtype), x0_fp32)
                else:
                    # Weight does not need a convert
                    result = ggml_matmul_src1_x_src0_t(x0, x_2d, self.weight_shape, self.qtype)
                    result = result.view(new_shape)
            # allreduce to combine partial results and add bias if necessary
            if self.mp_group is not None:
                # TODO: implement for CPU logic for vLLM tp
                # deepspeed distibuted mode
                from deepspeed import comm as dist
                dist.inference_all_reduce(result, group=self.mp_group)
            if self.bias is not None:
                result += self.bias
        return result.to(x.dtype)


class FP16Linear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True,
                 mp_group=None, weight_type=1, optimize_lm_head=False):
        super().__init__(input_features, output_features, bias)
        self.in_len = input_features
        self.out_len = output_features
        self.weight_shape = (self.out_len, self.in_len)
        self.weight_length = self.out_len * self.in_len
        self.qtype = ggml_tensor_qtype["fp16"]
        self.mp_group = mp_group
        # weight_type = 1 means original weight
        # weight_type = 2 means weight has been transposed
        # weight_type = 3 means weight has been transposed by esimd method
        self.weight_type = 1
        self.optimize_lm_head = optimize_lm_head
        self.disable_fp16_opt = False

    def forward(self, x: torch.Tensor):
        # only work for GPU
        invalidInputError(x.device.type == "xpu",
                          "FP16Linear only works for Intel GPUs")
        if self.optimize_lm_head:
            x = reshape_lm_head_input(x)

        x = x.to(torch.float16)
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        if self.weight is not None and self.weight.dtype != x.dtype:
            self.weight.data = self.weight.data.to(x.dtype)

        if not self.use_esimd_kernel(x):
            invalidInputError(self.weight_type == 1, "weight_type should be 1")
            result = F.linear(x, self.weight, self.bias)

            if self.mp_group is not None:
                if get_use_vllm():
                    result = self.mp_group.all_reduce(result)
                elif is_deepspeed_available():
                    from deepspeed import comm as dist
                    dist.inference_all_reduce(result, group=self.mp_group)
                else:
                    invalidInputError(False, "mp_group is not None, but no supported backend found")
            return result
        else:
            if self.in_len == 4096 and self.weight_type != 3 or \
                    self.in_len == 11008 and self.weight_type != 1:
                # convert weight first to use esimd fp16 kernel
                self.convert_weight_for_esimd_kernel()
            # esimd fp16 kernel for inference
            x_shape = x.shape
            x_2d = x.view(-1, x_shape[-1])
            if x_2d.is_contiguous() is False:
                x_2d = x_2d.contiguous()

            if x_2d.shape[0] > 8:
                # first token or batch size > 8, re-convert weight
                if self.weight_type == 3:
                    original_weight = self.weight.data.transpose(1, 2)
                    original_weight = original_weight.reshape(self.out_len, self.in_len)
                    result = F.linear(x_2d, original_weight.contiguous())
                    del original_weight
                else:
                    result = F.linear(x_2d, self.weight)
            else:
                # batch size <= 8, use esimd optimization
                import xe_batch
                result = xe_batch.batch_forward(x_2d, self.weight.data,
                                                self.qtype)

            new_shape = x_shape[:-1] + (self.out_len,)
            result = result.view(new_shape)
            if self.mp_group is not None:
                if get_use_vllm():
                    result = self.mp_group.all_reduce(result)
                elif is_deepspeed_available():
                    from deepspeed import comm as dist
                    dist.inference_all_reduce(result, group=self.mp_group)
                else:
                    invalidInputError(False, "mp_group is not None, but no supported backend found")
            if self.bias is not None:
                result += self.bias

            return result.to(x.dtype)

    def use_esimd_kernel(self, x):
        gpu_type = get_xpu_device_name(x.device)
        if self.disable_fp16_opt:
            return False
        # esimd kernel can only be used for Arc and Flex
        if gpu_type not in ["arc"]:
            return False
        # now esimd kernel can only be used for specific cases (llama2-7b shape)
        if self.in_len == 11008 and self.out_features == 4096:
            return True
        if self.in_len == 4096 and self.out_features in [4096, 11008]:
            # seems has some issue with Mistral,
            # need a further look to check whether can be used for other out features
            return True
        return False

    def convert_weight_for_esimd_kernel(self):
        m, n = self.out_len, self.in_len
        if self.in_len == 11008:
            if self.weight_type == 2:
                trans_weight = self.weight.data.transpose(0, 1)
            else:
                trans_weight = self.weight.data
            self.weight.data = trans_weight.contiguous()
            self.weight_type = 1
        elif self.in_len == 4096:
            if self.weight_type == 2:
                trans_weight = self.weight.data.transpose(0, 1)
            else:
                trans_weight = self.weight.data
            trans_weight = trans_weight.data.reshape(m//16, 16, n)
            trans_weight = trans_weight.transpose(1, 2).contiguous()
            self.weight.data = trans_weight
            self.weight_type = 3


class BF16Linear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True,
                 mp_group=None, compute_dtype=None, optimize_lm_head=False):
        super().__init__(input_features, output_features, bias)
        self.in_len = input_features
        self.out_len = output_features
        self.weight_shape = (self.out_len, self.in_len)
        self.weight_length = self.out_len * self.in_len
        self.qtype = ggml_tensor_qtype["bf16"]
        self.mp_group = mp_group
        self.compute_dtype = compute_dtype
        self.optimize_lm_head = optimize_lm_head

    def forward(self, x: torch.Tensor):
        if self.optimize_lm_head:
            x = reshape_lm_head_input(x)

        x = x.to(torch.bfloat16)
        if self.weight is not None and self.weight.dtype != x.dtype:
            self.weight.data = self.weight.data.to(x.dtype)
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # If x.shape>3, F.linear will use bmm, accounting for performance degradation.
        original_shape = x.shape
        # Convert to 2D shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        result = F.linear(x, self.weight, self.bias)

        # Convert to original shape
        if len(original_shape) > 2:
            result = result.reshape(*original_shape[:-1], result.shape[-1])

        return result.to(x.dtype)


class vLLMLowBitLinear(LowBitLinear):
    def __init__(self, input_features, output_features, qtype, bias=True,
                 conver_to_half=True, mp_group=None,
                 optimize_lm_head=False, act_order=False,
                 enable_scale_search=False):
        super().__init__(input_features, output_features, qtype, bias, conver_to_half, mp_group,
                         optimize_lm_head, act_order, enable_scale_search)

    def forward(self, x: torch.Tensor):
        result = super().forward(x)
        return result, None


class vLLMFP16Linear(FP16Linear):
    def __init__(self, input_features, output_features, bias=True, mp_group=None, weight_type=1,
                 optimize_lm_head=False):
        super().__init__(input_features, output_features, bias, mp_group, weight_type,
                         optimize_lm_head)

    def forward(self, x: torch.Tensor):
        result = super().forward(x)
        return result, None


class vLLMBF16Linear(BF16Linear):
    def __init__(self, input_features, output_features, bias=True, mp_group=None,
                 compute_dtype=None, optimize_lm_head=False):
        super().__init__(input_features, output_features, bias, mp_group, compute_dtype,
                         optimize_lm_head)

    def forward(self, x: torch.Tensor):
        result = super().forward(x)
        return result, None

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
# This file is adapted from
# https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear.py
#
# MIT License
#
# Copyright (c) 2023 MIT HAN Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import torch
import torch.nn as nn
from bigdl.llm.utils.common import invalidOperationError, invalidInputError
from transformers import AwqConfig
from transformers.utils.quantization_config import AwqBackendPackingMethod


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        invalidOperationError(False,
                              f"Not implemented group size {group_size}.")

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


class WQLinear_GEMM(nn.Module):
    def __init__(self, bits, group_size, in_features, out_features, bias, dev, backend):
        super().__init__()

        invalidOperationError(bits == 4, "Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        self.backend = backend

        # quick sanity check (make sure aligment)
        invalidInputError(self.in_features % self.group_size == 0,
                          f"Invalid in_features number {self.in_features}.")
        invalidInputError(out_features % (32 // self.bits) == 0,
                          f"Invalid out_features number {out_features}.")
        if backend == AwqBackendPackingMethod.LLMAWQ:
            self.wf = (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                                    dtype=torch.int32) * self.bits).unsqueeze(0)
            self.register_buffer('qweight',
                                 torch.zeros((out_features,
                                              in_features // (32 // self.bits)),
                                             dtype=torch.int32, device=dev))
            zeros_width = calculate_zeros_width(in_features, self.group_size)
            self.register_buffer('qzeros',
                                 torch.zeros((out_features, zeros_width),
                                             dtype=torch.int32, device=dev))
            self.register_buffer('scales',
                                 torch.zeros((out_features, zeros_width * (32 // self.bits)),
                                             dtype=torch.float16, device=dev))
        elif backend == AwqBackendPackingMethod.AUTOAWQ:
            self.wf = (torch.tensor([0, 4, 1, 5, 2, 6, 3, 7],
                                    dtype=torch.int32) * self.bits).unsqueeze(0)
            self.register_buffer('qweight',
                                 torch.zeros((in_features,
                                              out_features // (32 // self.bits)),
                                             dtype=torch.int32, device=dev))
            self.register_buffer('qzeros',
                                 torch.zeros((in_features // self.group_size,
                                              out_features // (32 // self.bits)),
                                             dtype=torch.int32, device=dev))
            self.register_buffer('scales',
                                 torch.zeros((in_features // self.group_size, out_features),
                                             dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16,
                                                     device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, bits, group_size, backend,
                    init_only=False, scales=None, zeros=None):
        awq_linear = cls(bits, group_size, linear.in_features, linear.out_features,
                         linear.bias is not None, linear.weight.device, backend)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        invalidInputError(scales is not None and zeros is not None,
                          "Scales and zeros should not be None.")
        scale_zeros = zeros * scales

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.bits

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round((linear.weight.data[:, idx] +
                             scale_zeros[idx // group_size]) /
                            awq_linear.scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0],
                               intweight.shape[1] // (32 // awq_linear.bits)),
                              dtype=torch.int32, device=intweight.device)

        torch.set_printoptions(threshold=10_000)
        print(intweight)

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.bits == 4:
                if backend == AwqBackendPackingMethod.AUTOAWQ:
                    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
                elif backend == AwqBackendPackingMethod.LLMAWQ:
                    order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                invalidOperationError(False, "Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.bits)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // (32 // awq_linear.bits)),
                             dtype=torch.int32, device=zeros.device)

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.bits == 4:
                if backend == AwqBackendPackingMethod.AUTOAWQ:
                    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
                elif backend == AwqBackendPackingMethod.LLMAWQ:
                    order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                invalidOperationError(False, "Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.bits)
        awq_linear.qzeros = qzeros

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        invalidOperationError(False, "Bigdl-llm does not support inference awq models directly.")

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, bits={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.bits, self.group_size
        )


class WQLinear_GEMV(nn.Module):
    def __init__(self, bits, group_size, in_features, out_features, bias, dev):
        super().__init__()

        invalidOperationError(bits == 4, "Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8

        # quick sanity check (make sure aligment)
        invalidInputError(self.in_features % self.group_size == 0,
                          f"Invalid in_features number {self.in_features}.")
        invalidInputError(out_features % (32 // self.bits) == 0,
                          f"Invalid out_features number {out_features}.")
        pack_num = (32 // self.bits)

        self.register_buffer('qweight',
                             torch.zeros((out_features, in_features // pack_num),
                                         dtype=torch.int32, device=dev))
        self.register_buffer('qzeros',
                             torch.zeros((out_features,
                                          calculate_zeros_width(in_features,
                                                                self.group_size)),
                                         dtype=torch.int32, device=dev))
        self.register_buffer('scales',
                             torch.zeros((out_features,
                                          calculate_zeros_width(in_features, self.group_size)
                                          * pack_num), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features),
                                                     dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, bits, group_size, backend,
                    init_only=False, scales=None, zeros=None):
        awq_linear = cls(bits, group_size, linear.in_features, linear.out_features,
                         linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        invalidInputError(scales is not None and zeros is not None,
                          "Scales and zeros should not be None.")
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.bits
        qscales = torch.zeros(
            (scales.shape[0], calculate_zeros_width(linear.in_features, group_size) * pack_num),
            dtype=torch.float16,
            device=scales.device
        )
        qscales[:, :scales.shape[1]] = scales
        awq_linear.scales = qscales
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round((linear.weight.data[:, idx] +
                             scale_zeros[:, idx // group_size]) /
                            awq_linear.scales[:, idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.bits),
                              dtype=torch.int32, device=intweight.device)

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.bits == 4:
                if backend == AwqBackendPackingMethod.AUTOAWQ:
                    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
                elif backend == AwqBackendPackingMethod.LLMAWQ:
                    order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                invalidOperationError(False, "Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.bits)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, group_size)),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range((zeros.shape[1] + pack_num - 1) // pack_num):
            if awq_linear.bits == 4:
                if backend == AwqBackendPackingMethod.AUTOAWQ:
                    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
                elif backend == AwqBackendPackingMethod.LLMAWQ:
                    order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                invalidOperationError(False, "Only 4-bit are supported for now.")
            for i in range(pack_num):
                if col * pack_num + order_map[i] >= zeros.shape[1]:
                    continue
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.bits)
        awq_linear.qzeros = qzeros
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        invalidOperationError(False, "Bigdl-llm does not support inference awq models directly.")

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, bits={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.bits, self.group_size
        )

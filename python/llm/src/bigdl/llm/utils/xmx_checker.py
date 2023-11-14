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

import torch
import intel_extension_for_pytorch as ipex
from bigdl.llm.ggml.quantize import ggml_tensor_qtype


SYM_INT4 = ggml_tensor_qtype["sym_int4"]
SYM_INT8 = ggml_tensor_qtype["sym_int8"]
NF4 = ggml_tensor_qtype["nf4"]
NF3 = ggml_tensor_qtype["nf3"]
FP8 = ggml_tensor_qtype["fp8"]
FP4 = ggml_tensor_qtype["fp4"]
MOFQ4 = ggml_tensor_qtype["mixed_fp4"]
MOFQ8 = ggml_tensor_qtype["mixed_fp8"]


class XMXChecker:
    def __init__(self):
        self.support_xmx = self.check_xmx()
        self.supported_qtype = [SYM_INT4, SYM_INT8, FP8]

    @staticmethod
    def check_xmx():
        name = torch.xpu.get_device_name(0)
        # todo: not sure how to check xmx or how to get device name for now
        return "Arc(TM)" in name or "GPU Max" in name or "GPU Flex" in name

    def check(self, input_tensor: torch.Tensor, qtype: int):
        return self.support_xmx and 1 < input_tensor.shape[0] <= 8 and \
            qtype in self.supported_qtype


xmx_checker = XMXChecker()


def use_xmx(input_tensor: torch.Tensor, qtype: int):
    return xmx_checker.check(input_tensor, qtype)

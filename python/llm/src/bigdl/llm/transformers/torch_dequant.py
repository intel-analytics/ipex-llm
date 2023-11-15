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

from bigdl.llm.transformers.low_bit_linear import NF4, SYM_INT4
import torch

def torch_dequant_int4(quant_weight, scales, original_shape=(4096, 4096), dtype=torch.bfloat16):
    quants = quant_weight
    quants = quants.reshape(-1, 32)
    quants_0 = quants.bitwise_right_shift(torch.tensor(4, dtype=torch.uint8, device="hpu"))
    quants_1 = quants.bitwise_and(torch.tensor(0x0f, dtype=torch.uint8, device="hpu"))

    quants = torch.cat([quants_1, quants_0], dim=-1)
    quants_int = quants.to(dtype) - torch.tensor(8, dtype=dtype, device="hpu")
    dequants = (quants_int.to(dtype)) * scales.reshape(-1, 1)

    return dequants.reshape(original_shape)

def torch_dequant(quant_weight, scales, qtype, original_shape=(4096, 4096), dtype=torch.bfloat16):
    if qtype == SYM_INT4:
        return torch_dequant_int4(quant_weight, scales, original_shape, dtype)
    else:
        raise ValueError(f"Unsupported qtype: {qtype}")
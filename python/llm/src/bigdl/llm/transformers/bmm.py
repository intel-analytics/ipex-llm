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
import linear_q4_0
torch_bmm_old_ = torch.bmm


def torch_bmm(a, b):
    if a.device.type == 'cpu':
        return torch_bmm_old_(a, b)

    batch, A_rows, common = a.size()
    B_cols = b.size(2)
    C = torch.empty((batch, A_rows, B_cols), device=a.device)
    if a.size(1) == 1:
        torch_bmm_old_(a, b, out=C)
    else:
        linear_q4_0.bmm(a.contiguous(), b.contiguous(), C)
    return C


class SafeBMM:
    def __init__(self):
        self._old_bmm = torch_bmm_old_

    def __enter__(self):
        torch.bmm = torch_bmm

    def __exit__(self, *args, **kwargs):
        torch.bmm = self._old_bmm

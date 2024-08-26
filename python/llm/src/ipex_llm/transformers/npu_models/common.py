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


import torch
from typing import List


def merge_linear(linears: List[torch.nn.Linear]) -> torch.nn.Linear:
    new_weight = torch.cat(list(linear.weight.data for linear in linears), dim=0)
    if linears[0].bias is not None:
        new_linear = torch.nn.Linear(0, 0, bias=True)
        new_bias = torch.cat(list(linear.bias.data for linear in linears), dim=0)
        new_linear.bias = torch.nn.Parameter(new_bias, requires_grad=False)
    else:
        new_linear = torch.nn.Linear(0, 0, bias=False)
    new_linear.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    new_linear.in_features = new_weight.size(1)
    new_linear.out_features = new_weight.size(0)
    return new_linear


def reshape_lm_head_input(x):
    if x.dim() > 3:
        x = x.reshape([-1, x.shape[-2], x.shape[-1]])
    shape = list(x.size())
    if shape[1] > 10:
        shape[1] = 1
        x = x[:, -1, :].view(shape)
    return x

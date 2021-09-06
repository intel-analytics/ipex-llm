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
from torch.nn import functional as F
from typing import Callable

Tensor = torch.Tensor
_cross_entropy = F.cross_entropy


def replace_torch_function(function_name: str, replace_func: Callable):
    setattr(torch.nn.functional, function_name, replace_func)


def workaround_cross_entropy(
    input: Tensor,
    target: Tensor,
    *args,
    **kwargs
) -> Tensor:
    input = input.cpu()
    target = target.cpu()
    return _cross_entropy(input, target, *args, **kwargs)


# Usage: append your target method and your own implements to  `replacement_dict`

# Apply ops replacements here
replacement_dict = {
    "cross_entropy": workaround_cross_entropy
}


def apply_torch_functional_replacement():
    for k, v in replacement_dict.items():
        replace_torch_function(k, v)

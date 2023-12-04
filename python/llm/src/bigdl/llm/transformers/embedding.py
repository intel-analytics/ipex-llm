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
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Parameter
from typing import Optional


# To prevent insufficient available memory when moving embedding from XPU back to CPU,
# we can pin the embedding to CPU if `cpu_embedding==True`.
class CPUPinnedParam(Parameter):
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is None:
            return super().to(*args, **kwargs)
        elif device.type == 'xpu':
            if convert_to_format is not None and self.dim() in (4, 5):
                return super().to('cpu', dtype,
                                  non_blocking, memory_format=convert_to_format)
            return super().to('cpu', dtype, non_blocking)
        return super().to(*args, **kwargs)


class LLMEmbedding(torch.nn.Embedding):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _weight: Optional[Tensor] = None,
                 _freeze: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq, sparse,
                         _weight, device, dtype)
        self.weight = CPUPinnedParam(self.weight.data, requires_grad=not _freeze)

    def forward(self, x: Tensor):
        return super().forward(x.to('cpu')).to(x.device)

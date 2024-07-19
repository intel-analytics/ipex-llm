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


import numpy
import torch
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from ipex_llm.transformers.low_bit_linear import FP4Params
from ipex_llm.utils.common import invalidInputError


# To prevent insufficient available memory when moving embedding from XPU back to CPU,
# we can pin the embedding to CPU if `cpu_embedding==True`.
class CPUPinnedParam(Parameter):
    # Overwrite the device attribute for CPUPinnedParam so that its device will be same as
    # the device for model.to(device);
    # With this device attribute, model.device will be same as the
    # the device for model.to(device) even with cpu_embedding==True
    @property
    def device(self):
        try:
            return self._device
        except AttributeError:
            return super().device

    @device.setter
    def device(self, to_device):
        self._device = to_device

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is None:
            return super().to(*args, **kwargs)
        elif device.type == 'xpu':
            self.device = device
            if convert_to_format is not None and self.dim() in (4, 5):
                return super().to('cpu', dtype,
                                  non_blocking, memory_format=convert_to_format)
            return super().to('cpu', dtype, non_blocking)
        return super().to(*args, **kwargs)


class CPUEmbedding(torch.nn.Embedding):
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
                 device=None,
                 dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq,
                         sparse, _weight, True, device, dtype)
        self.weight = CPUPinnedParam(self.weight.data, requires_grad=False)

    def forward(self, x: Tensor):
        return super().forward(x.to('cpu')).to(x.device)

    @classmethod
    def from_embedding(cls, embedding: torch.nn.Embedding):
        return cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            embedding.padding_idx,
            embedding.max_norm,
            embedding.norm_type,
            embedding.scale_grad_by_freq,
            embedding.sparse,
            embedding.weight.data,
            True,
            embedding.weight.device,
            embedding.weight.dtype,
        )


class DiskEmbedding(torch.nn.Embedding):
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
                 device=None,
                 dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq,
                         sparse, _weight, True, device, dtype)
        self.filename = "embeddings.bin"
        self.weight.data.flatten().to(device='cpu', dtype=torch.half).numpy().tofile(self.filename)
        dummy_weight = torch.empty(0, 0, dtype=self.weight.dtype, device=self.weight.device)
        self.weight = torch.nn.Parameter(dummy_weight, requires_grad=False)

    def forward(self, input_ids: Tensor):
        ids = input_ids.cpu().flatten()

        embeds = []
        with open(self.filename, 'rb') as f:
            for idx in ids:
                f.seek(idx * self.embedding_dim * 2)
                buffer = f.read(self.embedding_dim * 2)
                embeds.append(torch.frombuffer(buffer, dtype=torch.half))
        embeds = torch.stack(embeds).to(device=input_ids.device, dtype=self.weight.dtype)
        return embeds.view(*input_ids.size(), self.embedding_dim)

    @classmethod
    def from_embedding(cls, embedding: torch.nn.Embedding):
        return cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            embedding.padding_idx,
            embedding.max_norm,
            embedding.norm_type,
            embedding.scale_grad_by_freq,
            embedding.sparse,
            embedding.weight.data,
            True,
            embedding.weight.device,
            embedding.weight.dtype,
        )

    def to_embedding(self):
        with open(self.filename, 'rb') as f:
            buffer = f.read()
        embeds = torch.frombuffer(buffer, dtype=torch.half).clone()
        embeds = embeds.view(self.num_embeddings, self.embedding_dim).to(
            device=self.weight.device, dtype=self.weight.dtype
        )
        return torch.nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
            embeds,
            True,
            embeds.device,
            embeds.dtype,
        )

    @staticmethod
    def replace_normal_embedding(m: torch.nn.Module):
        for name, module in m.named_children():
            if type(module) == torch.nn.Embedding:
                m._modules[name] = DiskEmbedding.from_embedding(module)

    @staticmethod
    def restore_normal_embedding(m: torch.nn.Module):
        for name, module in m.named_children():
            if type(module) == DiskEmbedding:
                m._modules[name] = module.to_embedding()


class LowBitEmbedding(torch.nn.Embedding):
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
                 device=None,
                 dtype=None,
                 convert_shape_only=None,
                 qtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq, sparse,
                         _weight, device, dtype)
        self.qweight = FP4Params(self.weight.data,
                                 requires_grad=False,
                                 quantized=False,
                                 _shape=None,
                                 convert_shape_only=convert_shape_only,
                                 qtype=qtype,
                                 in_features=embedding_dim)
        # this dummy_weight is used to record model's dtype and device
        dummy_weight = torch.empty(0, 0, dtype=self.weight.dtype, device=self.weight.device)
        self.weight = torch.nn.Parameter(dummy_weight, requires_grad=False)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

    def forward(self, x: Tensor):
        invalidInputError(x.device.type == "xpu",
                          "`LowBitEmbedding` only supports GPU now.")
        try:
            import xe_linear
        except ModuleNotFoundError:
            invalidInputError(False,
                              "Please `pip install bigdl_core_xe_21` first.")

        result = xe_linear.dequantize_rows(x.contiguous(), self.qweight.data,
                                           self.qweight.qtype, self.embedding_dim,
                                           self.num_embeddings)
        return result.to(self.weight.dtype)

    @classmethod
    def from_embedding(cls, embedding: torch.nn.Embedding, convert_shape_only, qtype):
        return cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            embedding.padding_idx,
            embedding.max_norm,
            embedding.norm_type,
            embedding.scale_grad_by_freq,
            embedding.sparse,
            embedding.weight.data,
            True,
            embedding.weight.device,
            embedding.weight.dtype,
            convert_shape_only,
            qtype,
        )

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
from typing import List, Optional, Tuple, Union

from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm._C import ops
from .model_convert import _model_mlp_convert, _model_attention_convert


def _ipex_llm_convert(model_runner):
    setattr(model_runner, "load_model", _ipex_llm_load_model)
    setattr(RotaryEmbedding, "forward", _ipex_llm_rotary_embedding_forward)
    setattr(RMSNorm, "forward", _ipex_llm_rmsnorm_forward)


def _ipex_llm_load_model(self) -> None:
    _model_mlp_convert()
    _model_attention_convert()

    self.model = get_model(self.model_config,
                           self.device_config,
                           lora_config=self.lora_config,
                           parallel_config=self.parallel_config,
                           scheduler_config=self.scheduler_config)

    from ipex_llm import optimize_model
    optimize_model(self.model, low_bit="bf16", torch_dtype=self.model_config.dtype)


def _ipex_llm_rotary_embedding_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    self.cos_sin_cache = self.cos_sin_cache.to(positions.device, dtype=query.dtype)
    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    if offsets is not None:
        ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                     self.cos_sin_cache,
                                     self.is_neox_style, self.rotary_dim,
                                     offsets)
    else:
        ops.rotary_embedding(positions, query, key, self.head_size,
                             self.cos_sin_cache, self.is_neox_style)
    return query, key


def _ipex_llm_rmsnorm_forward(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        ops.fused_add_rms_norm(
            x,
            residual,
            self.weight.data,
            self.variance_epsilon,
        )
        return x, residual
    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        self.weight.data,
        self.variance_epsilon,
    )
    return out

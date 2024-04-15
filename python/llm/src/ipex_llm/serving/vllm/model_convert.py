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

from vllm.model_executor.models.commandr import CohereMLP, CohereAttention
from vllm.model_executor.models.deepseek import DeepseekMLP, DeepseekAttention
from vllm.model_executor.models.gemma import GemmaMLP, GemmaAttention
from vllm.model_executor.models.gpt_j import GPTJMLP, GPTJAttention
from vllm.model_executor.models.llama import LlamaMLP, LlamaAttention
from vllm.model_executor.models.gemma import GemmaMLP, GemmaAttention
from vllm.model_executor.models.orion import OrionMLP, OrionAttention
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeAttention
from vllm.model_executor.models.qwen2 import Qwen2MLP, Qwen2Attention
from vllm.model_executor.models.stablelm import StablelmMLP, StablelmAttention
from vllm.model_executor.models.starcoder2 import Starcoder2MLP, Starcoder2Attention
from vllm.model_executor.models.xverse import XverseMLP, XverseAttention

from vllm.attention import AttentionMetadata

_MLP_CONVERT_LIST = [CohereMLP, DeepseekMLP, GemmaMLP, GPTJMLP, LlamaMLP,
                     OrionMLP, Qwen2MoeMLP, Qwen2MLP, StablelmMLP, Starcoder2MLP,
                     XverseMLP]

_ATTENTION_CONVERT_LIST = [CohereAttention, DeepseekAttention, GemmaAttention, GPTJAttention,
                           LlamaAttention, OrionAttention,
                           Qwen2MoeAttention, Qwen2Attention, StablelmAttention,
                           Starcoder2Attention, XverseAttention]


def _MLP_forward(self, x):
    gate_up = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x = self.down_proj(x)
    return x


def _Attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    kv_cache = kv_cache.to(dtype=k.dtype)
    if hasattr(self, "kv_scale"):
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                self.kv_scale)
    else:
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output = self.o_proj(attn_output)
    return output


def _model_mlp_convert():
    for mlp in _MLP_CONVERT_LIST:
        setattr(mlp, "forward", _MLP_forward)


def _model_attention_convert():
    for attention in _ATTENTION_CONVERT_LIST:
        setattr(attention, "forward", _Attention_forward)

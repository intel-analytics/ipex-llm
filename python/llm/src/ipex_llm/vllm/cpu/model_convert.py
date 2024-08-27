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
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.model_executor.models.llama import LlamaMLP, LlamaAttention
from vllm.model_executor.models.qwen2 import Qwen2MLP, Qwen2Attention
from vllm.model_executor.models.qwen import QWenMLP, QWenAttention
from vllm.model_executor.models.baichuan import BaiChuanMLP, BaiChuanAttention
from vllm.model_executor.models.chatglm import GLMMLP, GLMAttention
from vllm.attention import Attention, AttentionMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.config import DeviceConfig
from vllm.logger import init_logger

from vllm._C import ops
from ipex_llm.utils.common import invalidInputError
from typing import List, Optional, Tuple, Union

logger = init_logger(__name__)


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
    qkv = self.qkv_proj(hidden_states).to(dtype=kv_cache.dtype)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale)
    output = self.o_proj(attn_output)
    return output


def _QWen_Attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.c_attn(hidden_states).to(dtype=kv_cache.dtype)
    q, k, v = qkv.chunk(chunks=3, dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output = self.c_proj(attn_output)
    return output


def _QWen_MLP_forward(self, x):
    gate_up = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x = self.c_proj(x)
    return x


def _Qwen2_Attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states).to(dtype=kv_cache.dtype)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output = self.o_proj(attn_output)
    return output


def _ChatGLM_MLP_forward(self, hidden_states):
    # [s, b, 4hp]
    intermediate_parallel = self.dense_h_to_4h(hidden_states)
    intermediate_parallel = self.activation_func(intermediate_parallel)
    # [s, b, h]
    output = self.dense_4h_to_h(intermediate_parallel)
    return output


def _Baichuan_Attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.W_pack(hidden_states).to(dtype=kv_cache.dtype)
    q, k, v = qkv.chunk(chunks=3, dim=-1)
    if self.postion_embedding != "ALIBI":
        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output = self.o_proj(attn_output)
    return output


def _ChatGLM_Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.query_key_value(hidden_states).to(dtype=kv_cache.dtype)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(position_ids, q, k)
    context_layer = self.attn(
        q,
        k,
        v,
        kv_cache,
        attn_metadata,
    )
    attn_output = self.dense(context_layer)
    return attn_output

_REPLACED_MLP_LAYERS = {
    LlamaMLP: _MLP_forward,
    Qwen2MLP: _MLP_forward,
    BaiChuanMLP: _MLP_forward,
    # QWenMLP: _QWen_MLP_forward,
    GLMMLP: _ChatGLM_MLP_forward
}

_REPLACED_ATTENTION_LAYERS = {
    LlamaAttention: _Attention_forward,
    Qwen2Attention: _Qwen2_Attention_forward,
    # QWenAttention: _QWen_Attention_forward,
    BaiChuanAttention: _Baichuan_Attention_forward,
    GLMAttention: _ChatGLM_Attention_forward
}

_IPEX_LLM_SUPPORTED_MODELS = [
    "LlamaForCausalLM",
    "BaichuanForCausalLM",
    "ChatGLMForCausalLM",
    "Qwen2ForCausalLM",
]


def _model_mlp_convert():
    for module, replaced_func in _REPLACED_MLP_LAYERS.items():
        setattr(module, "forward", replaced_func)


def _model_attention_convert():
    for module, replaced_func in _REPLACED_ATTENTION_LAYERS.items():
        setattr(module, "forward", replaced_func)


def _ipex_llm_convert(load_in_low_bit):
    if load_in_low_bit is None:
        return
    from vllm.worker.cpu_model_runner import CPUModelRunner
    import vllm.model_executor.model_loader as model_loader
    setattr(CPUModelRunner, "load_model", get_load_function(load_in_low_bit))

    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    setattr(RotaryEmbedding, "forward", _ipex_llm_rotary_embedding_forward)
    from vllm.model_executor.layers.layernorm import RMSNorm
    setattr(RMSNorm, "forward", _ipex_llm_rmsnorm_forward)


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
    x = x.to(dtype=self.weight.data.dtype)
    if residual is not None:
        residual = residual.to(dtype=self.weight.data.dtype)
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


def get_load_function(low_bit):
    def _ipex_llm_load_model(self) -> None:
        model_class = get_model_architecture(self.model_config)[1]
        cur_model_list = ", ".join(_IPEX_LLM_SUPPORTED_MODELS)
        if low_bit != "bf16":
            invalidInputError(model_class in _IPEX_LLM_SUPPORTED_MODELS,
                              f"Currently IPEX-LLM vLLM convert only support {cur_model_list}.")
        else:
            if model_class not in _IPEX_LLM_SUPPORTED_MODELS:
                logger.warning(
                    f"Currently IPEX-LLM vLLM convert only support {cur_model_list}."
                )
                self.model = get_model(
                    model_config=self.model_config,
                    load_config=self.load_config,
                    device_config=self.device_config,
                    vision_language_config=self.vision_language_config,
                    lora_config=self.lora_config,
                    parallel_config=self.parallel_config,
                    scheduler_config=self.scheduler_config)
                return

        # _model_mlp_convert()
        # _model_attention_convert()

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
            vision_language_config=self.vision_language_config,
            lora_config=self.lora_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config)

        from ipex_llm import optimize_model
        optimize_model(self.model, low_bit=low_bit, torch_dtype=self.model_config.dtype)

    return _ipex_llm_load_model

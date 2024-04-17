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
from vllm.model_executor.models.llama import LlamaMLP, LlamaAttention
from vllm.model_executor.models.qwen2 import Qwen2MLP, Qwen2Attention
from vllm.model_executor.models.qwen import QWenMLP, QWenAttention
from vllm.model_executor.models.baichuan import BaiChuanMLP, BaiChuanAttention
from vllm.model_executor.models.chatglm import GLMMLP, GLMAttention
from vllm.model_executor.model_loader import get_model
from vllm.utils import measure_device_memory
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.utils import measure_device_memory
from vllm.model_executor.input_metadata import InputMetadata
from vllm.config import DeviceConfig
from typing import Tuple
from ipex_llm.utils.common import invalidInputError


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
    input_metadata: InputMetadata,
) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    k_cache, v_cache = kv_cache
    attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
    output = self.o_proj(attn_output)
    return output


def _QWen_Attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    input_metadata: InputMetadata,
) -> torch.Tensor:
    qkv = self.c_attn(hidden_states)
    q, k, v = qkv.chunk(chunks=3, dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    k_cache, v_cache = kv_cache
    attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
    output = self.c_proj(attn_output)
    return output


def _QWen_MLP_forward(self, x):
    gate_up = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x = self.c_proj(x)
    return x


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
    input_metadata: InputMetadata,
) -> torch.Tensor:
    qkv = self.W_pack(hidden_states)
    q, k, v = qkv.chunk(chunks=3, dim=-1)
    if self.postion_embedding != "ALIBI":
        q, k = self.rotary_emb(positions, q, k)
    k_cache, v_cache = kv_cache
    attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
    output = self.o_proj(attn_output)
    return output


def _ChatGLM_Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    input_metadata: InputMetadata,
) -> torch.Tensor:
    qkv = self.query_key_value(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(position_ids, q, k)
    key_cache, value_cache = kv_cache
    context_layer = self.attn(
        q,
        k,
        v,
        key_cache,
        value_cache,
        input_metadata,
    )
    attn_output = self.dense(context_layer)
    return attn_output

_REPLACED_MLP_LAYERS = {
    LlamaMLP: _MLP_forward,
    Qwen2MLP: _MLP_forward,
    BaiChuanMLP: _MLP_forward,
    QWenMLP: _QWen_MLP_forward,
    GLMMLP: _ChatGLM_MLP_forward
}

_REPLACED_ATTENTION_LAYERS = {
    LlamaAttention: _Attention_forward,
    Qwen2Attention: _Attention_forward,
    QWenAttention: _QWen_Attention_forward,
    BaiChuanAttention: _Baichuan_Attention_forward,
    GLMAttention: _ChatGLM_Attention_forward
}


def _model_mlp_convert():
    for module, replaced_func in _REPLACED_MLP_LAYERS.items():
        setattr(module, "forward", replaced_func)


def _model_attention_convert():
    for module, replaced_func in _REPLACED_ATTENTION_LAYERS.items():
        setattr(module, "forward", replaced_func)


def _ipex_llm_convert(load_in_low_bit):
    from vllm.worker.model_runner import ModelRunner
    import vllm.model_executor.model_loader as model_loader
    setattr(ModelRunner, "load_model", get_load_function(load_in_low_bit))


def get_load_function(low_bit):
    def _ipex_llm_load_model(self) -> None:
        _model_mlp_convert()
        _model_attention_convert()

        with measure_device_memory() as m:
            # only support xpu for now
            # We have to create a new DeviceConfig.
            # Otherwise, will get the wrong xpu memory usage
            self.model = get_model(self.model_config,
                                   DeviceConfig("cpu"),
                                   lora_config=self.lora_config,
                                   parallel_config=self.parallel_config,
                                   scheduler_config=self.scheduler_config)
            from ipex_llm import optimize_model
            optimize_model(self.model, low_bit=low_bit, torch_dtype=self.model_config.dtype)
            self.model = self.model.to(device=self.device_config.device,
                                       dtype=self.model_config.dtype)

        self.model_memory_usage = m.consumed_memory
        logger = init_logger(__name__)
        logger.info(f"Loading model weights took "
                    f"{self.model_memory_usage / float(2**30):.4f} GB")

        if self.lora_config:
            invalidInputError(hasattr(self.model, "supported_lora_modules")
                              and self.model.supported_lora_modules,
                              "Model does not support LoRA")
            invalidInputError(hasattr(self.model, "embedding_modules"),
                              "Model does not have embedding_modules")
            invalidInputError(hasattr(self.model, "embedding_padding_modules"),
                              "Model does not have embedding_padding_modules")
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens +
                self.scheduler_config.max_paddings, self.vocab_size,
                self.lora_config, self.device, self.model.embedding_modules,
                self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)
    return _ipex_llm_load_model

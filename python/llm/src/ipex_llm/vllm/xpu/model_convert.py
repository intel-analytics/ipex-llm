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
from typing import Optional
from vllm.distributed import tensor_model_parallel_gather
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaMLP, LlamaAttention, LlamaForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2MLP, Qwen2Attention, Qwen2ForCausalLM
from vllm.model_executor.models.qwen import QWenMLP, QWenAttention, QWenLMHeadModel
from vllm.model_executor.models.baichuan import BaiChuanMLP, BaiChuanAttention
from vllm.model_executor.models.baichuan import BaiChuanBaseForCausalLM
from vllm.model_executor.models.chatglm import GLMMLP, GLMAttention, ChatGLMForCausalLM
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.attention import AttentionMetadata
from vllm.config import DeviceConfig
from typing import Tuple


def _sample_get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    embedding_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    # HINT: we do not support other types of quantization for now
    logits = lm_head(hidden_states)
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_gather(logits)
    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, : self.org_vocab_size]
    return logits


def _model_sample_convert():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    setattr(LogitsProcessor, "_get_logits", _sample_get_logits)


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
    # k_cache, v_cache = kv_cache
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output = self.o_proj(attn_output)
    return output


def _QWen_Attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.c_attn(hidden_states)
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
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.W_pack(hidden_states)
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
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv = self.query_key_value(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(position_ids, q, k)
    # key_cache, value_cache = kv_cache
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

# _REPLACED_SAMPLER_LAYERS = {
#     LlamaForCausalLM: _Llama_sample,
#     QWenLMHeadModel: _Llama_sample,
#     ChatGLMForCausalLM: _Chatglm_sample,
#     Qwen2ForCausalLM: _Qwen2_sample,
#     BaiChuanBaseForCausalLM: _Llama_sample,
# }


def _model_mlp_convert():
    for module, replaced_func in _REPLACED_MLP_LAYERS.items():
        setattr(module, "forward", replaced_func)


# def _model_sample_convert():
#     setattr(Sampler, "_get_logits", _sample_get_logits)
#     for module, replaced_func in _REPLACED_SAMPLER_LAYERS.items():
#         setattr(module, "sample", replaced_func)


def _model_attention_convert():
    for module, replaced_func in _REPLACED_ATTENTION_LAYERS.items():
        setattr(module, "forward", replaced_func)


def _ipex_llm_convert(load_in_low_bit):
    from vllm.worker.xpu_model_runner import XPUModelRunner
    from ipex_llm.vllm.xpu.ipex_llm_wrapper import get_ipex_llm_wrapper
    import vllm.executor.ray_utils as ray_utils
    setattr(XPUModelRunner, "load_model", get_load_function(load_in_low_bit))
    setattr(ray_utils, "RayWorkerWrapper", get_ipex_llm_wrapper(load_in_low_bit))

from vllm.model_executor.models.utils import _CPU_OFFLOAD_MAX_BYTES
print(f"global: {_CPU_OFFLOAD_MAX_BYTES}")
_CPU_OFFLOAD_MAX_BYTES = 1 * 1024**3
_CPU_OFFLOAD_BYTES = 0
from vllm.utils import is_pin_memory_available
from torch.func import functional_call
def maybe_not_offload_to_xpu(module: torch.nn.Module) -> torch.nn.Module:
    device = "xpu"
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    offloaded_parameters = False

    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        module.to("xpu")
    else:
        total_memory = sum(p.numel() * p.element_size() for p in module.parameters())
        _CPU_OFFLOAD_BYTES = _CPU_OFFLOAD_BYTES + total_memory
        module.to("cpu")
        offloaded_parameters = True

    if offloaded_parameters:
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_state = {
                k: v.to(device)
                for k, v in module.named_parameters()
            }
            output = functional_call(module,
                                     device_state,
                                     args=args,
                                     kwargs=kwargs)
            module.forward = forward
            return output

        module.forward = forward

    return module


def get_load_function(low_bit):
    def _ipex_llm_load_model(self) -> None:
        _model_mlp_convert()
        _model_attention_convert()
        _model_sample_convert()

        # from vllm.utils import measure_device_memory
        from vllm.utils import CudaMemoryProfiler
        with CudaMemoryProfiler() as m:
            self.model = get_model(
                model_config=self.model_config,
                device_config=DeviceConfig("cpu"),
                load_config=self.load_config,
                lora_config=self.lora_config,
                multimodal_config=self.multimodal_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                cache_config=self.cache_config,
            )
            from ipex_llm import optimize_model
            optimize_model(self.model, low_bit=low_bit, torch_dtype=self.model_config.dtype)
            if False:
                self.model = self.model.to(device=self.device_config.device,
                                           dtype=self.model_config.dtype)
            else:
                # need to set commonly
                self.model.model.embed_tokens.to(self.device_config.device)
                self.model.model.norm.to(self.device_config.device)
                self.model.logits_processor.to(self.device_config.device)
                self.model.lm_head.to(self.device_config.device)
                self.model.sampler.to(self.device_config.device)
                for layer in self.model.model.layers:
                    maybe_not_offload_to_xpu(layer)
                    #layer.to(self.device_config.device)
            print(self.model)
            for name, param in self.model.model.layers[0].named_parameters():
                print(f"Parameter name: {name}, param: {param}")


        self.model_memory_usage = m.consumed_memory
        logger = init_logger(__name__)
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    return _ipex_llm_load_model


def _sample_get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    embedding_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    # HINT: we do not support other types of quantization for now
    logits = lm_head(hidden_states)
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_gather(logits)
    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, : self.org_vocab_size]
    return logits

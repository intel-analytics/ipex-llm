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
# Some parts of this file is adapted from
# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

import math
import torch
import warnings
from torch import nn

from ipex_llm.transformers.models.utils import (
    rotate_half, should_use_fuse_rope,
)
from ipex_llm.transformers.models.utils import mlp_fusion_check, SILU
from ipex_llm.transformers.models.utils import use_sdp, use_sdp_causal
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, restore_fp8_kv_cache
from ipex_llm.transformers.kv import DynamicNormalCache, DynamicFp8Cache

from typing import Optional, Tuple, List
from transformers.models.phi.modeling_phi import repeat_kv
from transformers.cache_utils import Cache


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def pre_compute_inv_freq(module: torch.nn.Module):
    if module.__class__.__name__ == "Phi3RotaryEmbedding":
        module.inv_freq = 1.0 / (
            module.base **
            (torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim)
        )
    elif module.__class__.__name__ == "Phi3SuScaledRotaryEmbedding":
        inv_freq_shape = torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim
        short_ext_factors = torch.tensor(module.short_factor, dtype=torch.float32)
        module.inv_freq = 1.0 / (short_ext_factors * module.base ** inv_freq_shape)

        long_ext_factors = torch.tensor(module.long_factor, dtype=torch.float32)
        module.register_buffer("long_inv_freq", None, persistent=False)
        module.long_inv_freq = 1.0 / (long_ext_factors * module.base ** inv_freq_shape)

        if module.max_position_embeddings <= module.original_max_position_embeddings:
            module.scaling_factor = 1.0
        else:
            scale = module.max_position_embeddings / module.original_max_position_embeddings
            module.scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(module.original_max_position_embeddings)
            )


def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    warnings.warn("You are not running the flash-attention implementation, "
                  "expect numerical differences.")

    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # IPEX-LLM OPT: fuse rope
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import linear_q4_0
        if self.rotary_emb.__class__.__name__ == "Phi3RotaryEmbedding":     # 4k
            linear_q4_0.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                             query_states, key_states)
        else:   # 128k
            if kv_seq_len > self.rotary_emb.original_max_position_embeddings:
                linear_q4_0.rotary_half_inplaced(self.rotary_emb.long_inv_freq, position_ids,
                                                 query_states, key_states)
            else:
                linear_q4_0.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                                 query_states, key_states)
            # todo: fuse scaling_factor
            query_states *= self.rotary_emb.scaling_factor
            key_states *= self.rotary_emb.scaling_factor
    else:
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)

    if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import linear_q4_0
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = linear_q4_0.sdp_fp8(query_states, key_states, value_states,
                                              attention_mask)
        else:
            attn_output = linear_q4_0.sdp(query_states, key_states, value_states, attention_mask)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import linear_q4_0
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = linear_q4_0.sdp_fp8_causal(query_states, key_states, value_states)
        else:
            attn_output = linear_q4_0.sdp_causal(query_states, key_states, value_states)
    else:
        if isinstance(past_key_value, DynamicFp8Cache):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                   dtype=torch.float32).to(value_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                                   training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def split_mlp(module: torch.nn.Module):
    if module.__class__.__name__ == "Phi3MLP":
        gate_weight, up_weight = module.gate_up_proj.weight.data.chunk(2, dim=0)

        gate_proj = torch.nn.Linear(0, 0, bias=False)
        gate_proj.weight = torch.nn.Parameter(gate_weight, requires_grad=False)
        gate_proj.in_features = gate_weight.size(1)
        gate_proj.out_features = gate_weight.size(0)

        up_proj = torch.nn.Linear(0, 0, bias=False)
        up_proj.weight = torch.nn.Parameter(up_weight, requires_grad=False)
        up_proj.in_features = up_weight.size(1)
        up_proj.out_features = up_weight.size(0)

        module.gate_proj = gate_proj
        module.up_proj = up_proj

        del module.gate_up_proj


def mlp_forward(
    self,
    hidden_states: torch.FloatTensor
) -> torch.FloatTensor:
    x_2d = hidden_states.view(-1, hidden_states.shape[-1])
    qtype = getattr(self.gate_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training):
        x_2d = x_2d.contiguous()
        import linear_q4_0
        return self.down_proj(linear_q4_0.mlp_forward_xpu(
            x_2d, self.gate_proj.weight.data, self.up_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.gate_proj.out_features,
            SILU, qtype
        ))
    return self.down_proj(
        self.activation_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
    )


def model_forward_wrapper(origin_model_forward):
    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # IPEX-LLM OPT: kv cache and quantize kv cache and sdp
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_quantize_kv = use_quantize_kv_cache(self.layers[0].mlp.down_proj, input_ids)
        if use_cache:
            if use_quantize_kv and not isinstance(past_key_values, DynamicFp8Cache):
                past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
            if not use_quantize_kv and not isinstance(past_key_values, DynamicNormalCache):
                past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
        return origin_model_forward(
            self=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    return model_forward


def phi3_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu" and not (self.training and hidden_states.requires_grad):
        import linear_q4_0
        x_2d = hidden_states.reshape(-1, hidden_states.size(-1)).contiguous()
        output = linear_q4_0.rms_norm(self.weight, x_2d, self.variance_epsilon)
        return output.reshape(hidden_states.shape)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

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

from ipex_llm.transformers.models.utils import (
    rotate_half, should_use_fuse_rope,
    apply_rotary_pos_emb_cache_freq_xpu
)
from ipex_llm.transformers.models.utils import mlp_fusion_check, SILU
from ipex_llm.transformers.models.utils import use_new_esimd_sdp_fp16, use_quantize_kv_cache
from ipex_llm.transformers.models.utils import use_sdp_fp8, restore_fp8_kv_cache
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


def su_scaled_rope_forward(self, x: torch.Tensor, position_ids: torch.Tensor, seq_len=None):
    if self.inv_freq is None:
        short_ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)
        inv_freq_shape = torch.arange(0, self.dim, 2,
                                      dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (short_ext_factors * self.base**inv_freq_shape)

        long_ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        self.register_buffer("long_inv_freq", None, persistent=False)
        self.long_inv_freq = 1.0 / (long_ext_factors * self.base**inv_freq_shape)

    seq_len = seq_len if seq_len is not None else torch.max(position_ids) + 1
    if seq_len > self.original_max_position_embeddings:
        inv_freq = self.long_inv_freq
    else:
        inv_freq = self.inv_freq

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    # Force float32 since bfloat16 loses precision on long contexts
    # See https://github.com/huggingface/transformers/pull/29285
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )

        cos = emb.cos() * scaling_factor
        sin = emb.sin() * scaling_factor
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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

    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    # IPEX-LLM OPT: fuse rope
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(query_states, key_states,
                                                                       sin, cos, "phi3")
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)

    if (isinstance(past_key_value, DynamicFp8Cache) and
            use_sdp_fp8(q_len, kv_seq_len, query_states)):
        import linear_q4_0
        attn_output = linear_q4_0.sdp_fp8(query_states, key_states, value_states, attention_mask)
    elif (isinstance(past_key_value, DynamicNormalCache) and
            use_new_esimd_sdp_fp16(q_len, kv_seq_len, self.head_dim, query_states)):
        import linear_q4_0
        attn_output = linear_q4_0.sdp_fp16(query_states, key_states, value_states, attention_mask)
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
        # IPEX-LLM OPT: kv cache but no sdp (its head_dim 96, cannot use sdp)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_quantize_kv = (use_quantize_kv_cache(self.layers[0].mlp.down_proj, input_ids) and
                           self.config.hidden_size // self.config.num_attention_heads in [64, 128])
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

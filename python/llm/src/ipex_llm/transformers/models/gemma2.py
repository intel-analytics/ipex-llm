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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

import torch

from typing import Optional, Tuple
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.utils import GELU
from ipex_llm.transformers.models.utils import should_use_fuse_rope, use_sdp, use_sdp_causal
from transformers.cache_utils import Cache
from transformers.models.gemma2.modeling_gemma2 import Gemma2Model, Gemma2Attention
from transformers.models.gemma2.modeling_gemma2 import repeat_kv, apply_rotary_pos_emb


def merge_qkv(module: torch.nn.Module):
    return merge_qkv_base(module, Gemma2Attention)


def gemma2_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    # ipex-llm change start: add kv_seq_len in past_key_values
    if past_key_values is not None:
        if cache_position is not None:
            kv_seq_len = cache_position[-1].item() + 1
        else:
            if input_ids is not None:
                kv_seq_len = input_ids.size(1)
            else:
                kv_seq_len = inputs_embeds.size(1)
        past_key_values.kv_seq_len = kv_seq_len
    # ipex-llm change end

    return Gemma2Model.forward(
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
        cache_position=cache_position
    )


def gemma2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    # IPEX-LLM OPT: fuse rope
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
        cos, sin = None, None
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "sliding_window": self.sliding_window,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, cache_kwargs)

    # IPEX_LLM OPT: sdp
    kv_seq_len = q_len if past_key_value is None else past_key_value.kv_seq_len
    if (use_sdp_causal(q_len, kv_seq_len, -1, query_states, self.training)
            and kv_seq_len <= key_states.size(2) and
            (self.sliding_window is None or kv_seq_len < self.sliding_window)):
        import xe_addons
        attn_weights = None
        attn_output = xe_addons.gemma2_sdp_causal(query_states,
                                                  key_states[:, :, :kv_seq_len, :],
                                                  value_states[:, :, :kv_seq_len, :],
                                                  attention_mask[:, :, :q_len, :kv_seq_len],
                                                  self.config.attn_logit_softcapping,
                                                  self.scaling)
    elif use_sdp(q_len, kv_seq_len, -1, query_states):
        import xe_addons
        attn_weights = None
        if self.sliding_window is not None:
            attn_mask = attention_mask[:, :, :q_len, : key_states.shape[-2]]
        else:
            attn_mask = attention_mask

        attn_output = xe_addons.gemma2_sdp(query_states,
                                           key_states,
                                           value_states,
                                           attn_mask,
                                           self.config.attn_logit_softcapping,
                                           self.scaling)
    else:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                   dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                                   training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

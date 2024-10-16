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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
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
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.transformers.models.common import merge_qkv_base, attention_softmax
from ipex_llm.transformers.models.utils import should_use_fuse_rope

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, repeat_kv
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding, GemmaAttention


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, GemmaAttention)


def pre_compute_inv_freq(module: torch.nn.Module):
    if isinstance(module, GemmaRotaryEmbedding):
        module.inv_freq = 1.0 / (
            module.base **
            (torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim)
        )


def gemma_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu" and not (self.training and hidden_states.requires_grad):
        import xe_addons
        x_2d = hidden_states.reshape(-1, hidden_states.size(-1)).contiguous()
        output = xe_addons.rms_norm(self.weight + 1, x_2d, self.eps)
        return output.reshape(hidden_states.shape)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
    return (1 + self.weight) * hidden_states.to(input_dtype)


def gemma_model_forward(
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
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    # IPEX-LLM OPT start: kv cache and quantize kv cache
    if use_cache and not isinstance(past_key_values, DynamicNormalCache):
        past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    # IPEX-LLM OPT end

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    invalidInputError((input_ids is None) ^ (inputs_embeds is None),
                      "You cannot specify both input_ids and inputs_embeds at the same time, "
                      "and must specify either one")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # IPEX-LLM changes start: support both transformers 4.38.1 and 4.39
    try:
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
        causal_mask = causal_mask[:, :, cache_position, :]
    except TypeError as _e:
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
    # IPEX-LLM changes end

    # embed positions
    hidden_states = inputs_embeds

    # normalized
    hidden_states = hidden_states * (self.config.hidden_size**0.5)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                     if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def gemma_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    cache_position: Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, None)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = attention_softmax(attn_weights, self.training)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import math
from typing import Optional, Tuple, Union, List

import torch

from ipex_llm.transformers.models.common import merge_qkv_base, attention_softmax
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, restore_fp8_kv_cache
from ipex_llm.transformers.models.utils import use_sdp, use_sdp_causal, should_use_fuse_rope
from ipex_llm.transformers.models.utils import use_sdp_non_causal
from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicNormalCache
from ipex_llm.utils.common import invalidInputError

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision
from transformers.models.qwen2_vl.modeling_qwen2_vl import repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, Qwen2VLAttention)


def qwen2_vl_model_forward(
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
    inputs = input_ids if input_ids is not None else inputs_embeds
    use_cache = True if inputs.device.type == "xpu" else use_cache
    use_quantize_kv = use_quantize_kv_cache(self.layers[0].mlp.down_proj, inputs)
    if use_cache:
        if use_quantize_kv and not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
        elif not use_quantize_kv and not isinstance(past_key_values, DynamicNormalCache):
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
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                                      device=inputs_embeds.device)

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # IPEX-LLM OPT start: use fused 2D rope
    if (torch.equal(position_ids[0], position_ids[1])
            and torch.equal(position_ids[0], position_ids[2])
            and should_use_fuse_rope(hidden_states, position_ids, False)):
        position_ids = position_ids[0].contiguous()
        position_embeddings = self.rotary_emb.inv_freq
    # IEPX_LLM OPT end

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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


def qwen2_vision_get_dtype(self) -> torch.dtype:
    return self.patch_embed.proj.weight.dtype


def qwen2_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1
                                              ).permute(1, 0, 2, 3).unbind(0)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    if len(cu_seqlens) == 2 and cu_seqlens.tolist() == [0, seq_length]:
        attention_mask = None
    else:
        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0

    if use_sdp_non_causal(self.head_dim, q.device, q.dtype):
        import xe_addons
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
        attn_output = xe_addons.sdp_non_causal(q, k.contiguous(), v.contiguous(), attention_mask)
        attn_output = attn_output.squeeze(0)
    else:
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attention_softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


def qwen2_vl_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    if position_ids.dim() == 2:
        import xe_addons
        inv_freq = position_embeddings
        xe_addons.rotary_half_inplaced(inv_freq, position_ids, query_states, key_states)
    else:
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)

    kv_seq_len = key_states.size(2)
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :kv_seq_len]

    attn_weights = None
    if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import xe_addons
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states, causal_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states, causal_mask)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, causal_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states,
                                               value_states, causal_mask)
    else:
        if isinstance(past_key_value, DynamicFp8Cache):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = attention_softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

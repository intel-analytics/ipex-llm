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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py

# coding=utf-8
# Copyright 2024 Cohere team. All rights reserved.
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

# This file is based on the LLama model definition file in transformers

"""PyTorch Cohere model."""
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List
from ipex_llm.transformers.models.llama import repeat_kv
from ipex_llm.transformers.models.utils import extend_kv_cache, append_kv_cache
from transformers.models.cohere.modeling_cohere import apply_rotary_pos_emb
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu,  is_enough_kv_cache_room_4_36
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.utils import use_decoding_fast_path
from ipex_llm.transformers.models.utils import use_flash_attention, use_esimd_sdp

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def cohere_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx)
    decoding_fast_path = use_decoding_fast_path(self.q_proj,
                                                True,
                                                enough_kv_room,
                                                bsz * q_len)
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k = past_key_value.key_cache[self.layer_idx]
        cache_v = past_key_value.value_cache[self.layer_idx]
        kv_seq_len = cache_k.shape[-2]
        import linear_q4_0
        query_states, key_states, value_states = linear_q4_0.forward_qkv(hidden_states,
                                                                         self.q_proj.weight,
                                                                         self.k_proj.weight,
                                                                         self.v_proj.weight,
                                                                         position_ids,
                                                                         cache_k, cache_v,
                                                                         self.q_proj.weight.qtype,
                                                                         self.v_proj.weight.qtype,
                                                                         kv_seq_len,
                                                                         self.head_dim,
                                                                         self.rotary_emb.base,)
        kv_seq_len += 1
        # update past_key_value's seem_tokens and kv caches.
        if self.layer_idx == 0:
            past_key_value._seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if self.layer_idx == 0:
                past_key_value._seen_tokens += key_states.shape[-2]

            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(key_states)
                past_key_value.value_cache.append(value_states)
            else:
                cache_k = past_key_value.key_cache[self.layer_idx]
                cache_v = past_key_value.value_cache[self.layer_idx]

                if not enough_kv_room:
                    # allocate new
                    new_c_k, new_c_v = extend_kv_cache(bsz,
                                                        self.num_key_value_heads,  # Support GQA
                                                        self.head_dim,
                                                        cache_k.size(2),
                                                        kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                        dtype=cache_k.dtype,
                                                        device=device)

                    new_c_k[:] = cache_k
                    new_c_v[:] = cache_v
                    cache_k = new_c_k
                    cache_v = new_c_v

                key_states, value_states = append_kv_cache(cache_k,
                                                            cache_v,
                                                            key_states,
                                                            value_states)

                # update past_key_value
                past_key_value.key_cache[self.layer_idx] = key_states
                past_key_value.value_cache[self.layer_idx] = value_states

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, attention_mask):
        attn_output = F.scaled_dot_product_attention(query_states.to(device, dtype=torch.float16),
                                                     key_states.to(device, dtype=torch.float16),
                                                     value_states.to(device, dtype=torch.float16),
                                                     is_causal=True)
        attn_weights = None
    elif not self.training and not hidden_states.requires_grad and \
            use_esimd_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import linear_q4_0
        attn_output = linear_q4_0.sdp_fp16(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
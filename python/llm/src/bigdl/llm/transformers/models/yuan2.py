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
# https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/main/yuan_hf_model.py
#
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
import copy
from einops import rearrange
from typing import Optional, Tuple

import torch
import torch.nn as nn
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import is_enough_kv_cache_room_4_31

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    before_hidden_states = None
    is_first_step = False

    enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value)

    if use_cache:
        if past_key_value is None:
            inference_hidden_states_memory = torch.empty(bsz, 2, hidden_states.shape[2], dtype=hidden_states.dtype)
            is_first_step = True
        else:
            before_hidden_states = past_key_value[2]

    if use_cache:
        if is_first_step:
            if q_len >= 2:
                inference_hidden_states_memory = hidden_states[ :, -2:, :]
            else:
                inference_hidden_states_memory[:, :, :] = 0
                inference_hidden_states_memory[:, -1:, :] = hidden_states[:, -1:, :]
        else:
            hidden_states_tmp = before_hidden_states[:, -1:, :]
            inference_hidden_states_memory = copy.deepcopy(torch.cat((hidden_states_tmp, hidden_states), dim=1))

    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    if self.use_shareqk:
        qk_states = self.qk_proj(hidden_states).view(bsz, q_len, self.num_heads*self.head_dim)
        query_key = qk_states.unsqueeze(2) * self.qk_weight + self.qk_bias
        query_states, key_states = torch.unbind(query_key, dim=2)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    else:
        hidden_states = self.lf_gate(hidden_states,before_hidden_states)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        qk_states = torch.cat([query_states, key_states], dim=-1)
        qk_states = qk_states.view(bsz,q_len,self.num_heads,int(qk_states.shape[-1]//self.num_heads))
        (query_states,key_states) =  torch.chunk(qk_states, 2, dim=-1)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids, "yuan2")

    if past_key_value is not None:
        # reuse k, v, self_attention
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if not enough_kv_room:
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.d_type,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        
        key_states, value_states = append_kv_cache(cache_k, cache_v, key_states, value_states)

    elif use_cache:
            max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_key_states, new_value_states = init_kv_cache(bsz,
                                                             self.num_heads,
                                                             self.head_dim,
                                                             kv_seq_len,
                                                             max_cache_length,
                                                             dtype=key_states.dtype,
                                                             device=device)
            new_key_states[:] = key_states
            new_value_states[:] = value_states
            key_states = new_key_states
            value_states = new_value_states

    past_key_value = (key_states, value_states, inference_hidden_states_memory) if use_cache else None

    if self.use_flash_attention:
        from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func
        attn_weights = None
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        batch_size, seqlen_q = query_states.shape[0], query_states.shape[1]
        seqlen_k = key_states.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [query_states, key_states, value_states]]

        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int,
                                device=q.device)
                                
        if self.training:                        
            assert seqlen_k == seqlen_q
            cu_seqlens_k = cu_seqlens_q
            is_causal = self.causal_mask
        else:
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int,
                    device=q.device)
            self.dropout=0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, self.dropout, causal=is_causal
        )

        attn_output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            invalidInputError(
                False,
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            invalidInputError(
                False,
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value
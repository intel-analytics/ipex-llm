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
# https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py
# which is licensed under Apache License 2.0:
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
from ipex_llm.transformers.models.utils import should_use_fuse_rope
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, \
    append_kv_cache, is_enough_kv_cache_room_4_31

import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def gptneox_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size]
    #   --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size: 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims:]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims:]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2]
    enough_kv_room = True
    if has_layer_past:
        enough_kv_room = is_enough_kv_cache_room_4_31(layer_past, seq_len=seq_len)
        seq_len += layer_past[0].shape[-2]

    use_fuse_rope = query.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and query.requires_grad)

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.maybe_rotary.inv_freq, position_ids,
                                       query_rot, key_rot)
        query = query_rot
        key = key_rot
    else:
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids, "gpt_neox")

    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # Cache QKV values
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        if not enough_kv_room:
            # allocate new
            new_past_key, new_past_value = extend_kv_cache(bsz,
                                                           self.num_attention_heads,
                                                           self.head_size,
                                                           past_key.size(2),
                                                           seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                           dtype=past_key.dtype,
                                                           device=device)
            new_past_key[:] = past_key
            new_past_value[:] = past_value
            past_key = new_past_key
            past_value = new_past_value

        key, value = append_kv_cache(past_key, past_value, key, value)
    elif use_cache:
        max_cache_length = seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        new_key, new_value = init_kv_cache(bsz,
                                           self.num_attention_heads,
                                           self.head_size,
                                           seq_len,
                                           max_cache_length,
                                           dtype=key.dtype,
                                           device=device)
        new_key[:] = key
        new_value[:] = value
        key = new_key
        value = new_value

    present = (key, value) if use_cache else None

    # Compute attention
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    # Reshape outputs
    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
    attn_output = self.dense(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs

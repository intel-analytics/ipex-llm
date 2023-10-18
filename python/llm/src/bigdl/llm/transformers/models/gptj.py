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
# This file is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
#

import torch
from typing import Optional, Tuple, Union
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, \
    apply_rotary_pos_emb, append_kv_cache
from transformers.utils.import_utils import is_torch_fx_proxy


KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def _get_embed_positions(self, position_ids):
    embed_positions = self.embed_positions
    if embed_positions.device != position_ids.device:
        embed_positions = embed_positions.to(position_ids.device)
        self.embed_positions = embed_positions
    return embed_positions.repeat(position_ids.shape[0], 1, 1)


def _attn(
    self,
    query,
    key,
    value,
    attention_mask=None,
    head_mask=None,
):
    # compute causal mask from causal mask buffer
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

    # Keep the attention weights computation in fp32 to avoid overflow issues
    query = query.to(torch.float32)
    key = key.to(torch.float32)

    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    mask_value = torch.finfo(attn_weights.dtype).min
    # Need to be a tensor, otherwise we get error:
    # `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    attn_weights = attn_weights / self.scale_attn

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def gptj_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Tuple[torch.Tensor]],
    Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
]:
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
    value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

    if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
        # The logic to conditionally copy to GPU could not be traced, so we do this
        # every time in the torch.fx case
        embed_positions = get_embed_positions(self.embed_positions, position_ids)
    else:
        embed_positions = self._get_embed_positions(position_ids)

    repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

    if self.rotary_dim is not None:
        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim:]

        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim:]

        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids, "gptj")

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
    else:
        query, key = apply_rotary_pos_emb(query, k_rot, cos, sin, position_ids, "gptj")

    batch_size, q_len, _ = hidden_states.size()

    key = key.permute(0, 2, 1, 3).contiguous()
    query = query.permute(0, 2, 1, 3).contiguous()

    kv_seq_len = key.size(-2)
    device = hidden_states.device

    if layer_past is not None:
        kv_seq_len += layer_past[0].size(1)

    if layer_past is not None:
        cache_k = layer_past[0]
        cache_v = layer_past[1]
        cache_k = cache_k.permute(0, 2, 1, 3)
        cache_v = cache_v.permute(0, 2, 1, 3)
        past_length = cache_k.size(2)

        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            new_cache_k, new_cache_v = extend_kv_cache(batch_size,
                                                       self.num_attention_heads,
                                                       self.head_dim,
                                                       past_length,
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        key, value = append_kv_cache(cache_k, cache_v, key, value)

    elif use_cache:
        key_cache, value_cache = init_kv_cache(batch_size,
                                               self.num_attention_heads,
                                               self.head_dim,
                                               kv_seq_len,
                                               kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                               dtype=key.dtype,
                                               device=device)
        key_cache[:] = key
        value_cache[:] = value
        key = key_cache
        value = value_cache

    if use_cache is True:
        present = (key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3))
    else:
        present = None

    # compute self-attention: V x Softmax(QK^T)
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)

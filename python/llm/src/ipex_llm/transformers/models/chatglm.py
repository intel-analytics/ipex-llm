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
# https://huggingface.co/THUDM/chatglm-6b/blob/63ce1bac4a7a7da57c67448bab39ddbe0e115a19/configuration_chatglm.py
#

import math
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from typing import Optional, Tuple
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import use_flash_attention, use_sdp


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


def glm_sdpa(query, key, value, attention_mask=None, is_causal=False):
    if use_flash_attention(query, key, attention_mask) or query.device.type == 'cpu':
        context_layer = F.scaled_dot_product_attention(query.to(key.dtype),
                                                       key,
                                                       value,
                                                       attention_mask,
                                                       is_causal=is_causal).to(key.dtype)
    else:
        # attention_mask is not None only when past_key_value is not None and q_len > 1
        if attention_mask is not None:
            attn_bias = torch.zeros(attention_mask.shape, dtype=query.dtype,
                                    device=query.device)
            attention_mask = ~attention_mask
            if attention_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attention_mask
        elif is_causal:
            L, S = query.size(-2), key.size(-2)
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(key.dtype)
        else:
            attn_bias = None
        if use_sdp(query.shape[2], key.shape[2],
                   query.shape[-1], query):
            import xe_addons
            attn_output = xe_addons.sdp(query, key, value, attn_bias)
            context_layer = attn_output.view(query.shape)
        else:
            head_dim = query.size(-1)
            attn = torch.matmul(query.to(key.dtype) / math.sqrt(head_dim),
                                key.transpose(2, 3))
            if attn_bias is not None:
                attn += attn_bias
            attn = F.softmax(attn, dim=-1,
                             dtype=torch.float32).to(value.dtype)
            context_layer = torch.matmul(attn, value)
    return context_layer


import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))
KV_CACHE_ALLOC_MIN_LENGTH = 512


def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        hidden_size_per_partition,
        layer_id,
        layer_past=None,
        scaling_attention_score=True,
        use_cache=False,
):
    key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
    value_layer = value_layer.permute(1, 2, 0, 3).contiguous()
    # query_layer = query_layer.permute(1, 2, 0, 3)

    cur_length, batch_size = query_layer.shape[0], query_layer.shape[1]
    device = query_layer.device

    if layer_past is not None:
        cache_k, cache_v = layer_past[0], layer_past[1]
        cache_k = cache_k.permute(1, 2, 0, 3)
        cache_v = cache_v.permute(1, 2, 0, 3)
        past_length = cache_k.size(2)
        if cache_k.stride()[1] < (past_length + cur_length) * cache_k.size(3):
            max_cache_length = past_length + cur_length + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_cache_k, new_cache_v = extend_kv_cache(batch_size,
                                                       self.num_attention_heads_per_partition,
                                                       self.hidden_size_per_attention_head,
                                                       past_length,
                                                       max_cache_length,
                                                       dtype=query_layer.dtype,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        key_layer, value_layer = append_kv_cache(cache_k, cache_v, key_layer, value_layer)

    elif use_cache:
        max_cache_length = max(KV_CACHE_ALLOC_MIN_LENGTH, cur_length) \
            + KV_CACHE_ALLOC_BLOCK_LENGTH
        key_cache, value_cache = init_kv_cache(batch_size, self.num_attention_heads_per_partition,
                                               self.hidden_size_per_attention_head, cur_length,
                                               max_cache_length,
                                               dtype=query_layer.dtype, device=device)
        key_cache[:] = key_layer
        value_cache[:] = value_layer
        key_layer = key_cache
        value_layer = value_cache

    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    b, nh, seq_len, hidden_size = key_layer.shape

    if use_cache:
        present = (key_layer.permute(2, 0, 1, 3), value_layer.permute(2, 0, 1, 3))
    else:
        present = None

    query_layer = query_layer.permute(1, 2, 0, 3)
    if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:

        if torch.is_autocast_cpu_enabled():
            attention_mask = torch.ones(query_layer.shape[2],
                                        key_layer.shape[2],
                                        dtype=torch.bool).tril(diagonal=0)
            attention_mask = attention_mask.masked_fill(~attention_mask, -float('inf'), )
            attention_mask = attention_mask.to(torch.get_autocast_cpu_dtype())
            query_layer = query_layer.to(torch.get_autocast_cpu_dtype())
            key_layer = key_layer.to(torch.get_autocast_cpu_dtype())
            value_layer = value_layer.to(torch.get_autocast_cpu_dtype())
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer,
                                                                             key_layer,
                                                                             value_layer,
                                                                             attention_mask,
                                                                             is_causal=False)
        else:
            context_layer = glm_sdpa(query_layer,
                                     key_layer,
                                     value_layer,
                                     attention_mask,
                                     is_causal=True)
    else:
        # attention_mask is not None only when past_key_value is not None and q_len > 1
        if attention_mask is not None:
            attn_bias = torch.zeros(attention_mask.shape, dtype=query_layer.dtype,
                                    device=query_layer.device)
            attention_mask = ~attention_mask
            if attention_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attention_mask
        else:
            attn_bias = None
        if torch.is_autocast_cpu_enabled():
            query_layer = query_layer.to(torch.get_autocast_cpu_dtype())
            key_layer = key_layer.to(torch.get_autocast_cpu_dtype())
            value_layer = value_layer.to(torch.get_autocast_cpu_dtype())
            attention_mask = attention_mask.to(torch.get_autocast_cpu_dtype())
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer,
                                                                             key_layer,
                                                                             value_layer,
                                                                             attention_mask)
        else:
            head_dim = query_layer.size(-1)
            attn = torch.matmul(query_layer.to(key_layer.dtype),
                                key_layer.transpose(2, 3)) / math.sqrt(head_dim)
            if attn_bias is not None:
                attn += attn_bias
            attn = F.softmax(attn, dim=-1,
                             dtype=torch.float32).to(value_layer.dtype)
            context_layer = torch.matmul(attn, value_layer)
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)
    attention_probs = None

    outputs = (context_layer, present, attention_probs)

    return outputs


def chatglm_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids,
        attention_mask: torch.Tensor,
        layer_id,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
        use_cache: bool = False,
        output_attentions: bool = False,
):
    """
    hidden_states: [seq_len, batch, hidden_size]
    attention_mask: [(1, 1), seq_len, seq_len]
    """

    # [seq_len, batch, 3 * hidden_size]
    mixed_raw_layer = self.query_key_value(hidden_states)

    # [seq_len, batch, 3 * hidden_size] -->
    #   [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
    new_tensor_shape = mixed_raw_layer.size()[:-1] + (
        self.num_attention_heads_per_partition,
        3 * self.hidden_size_per_attention_head,
    )
    mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)
    # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
    (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

    if self.position_encoding_2d:
        q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
        k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
        cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
        position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
            position_ids[:, 1, :].transpose(0, 1).contiguous()
        q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
        q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
        query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
        key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
    else:
        position_ids = position_ids.transpose(0, 1)
        cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer,
                                                            cos, sin, position_ids)

    # [seq_len, batch, hidden_size]
    context_layer, present, attention_probs = attention_fn(
        self=self,
        query_layer=query_layer,
        key_layer=key_layer,
        value_layer=value_layer,
        attention_mask=attention_mask,
        hidden_size_per_partition=self.hidden_size_per_partition,
        layer_id=layer_id,
        layer_past=layer_past,
        use_cache=use_cache
    )

    output = self.dense(context_layer)

    outputs = (output, present)

    if output_attentions:
        outputs += (attention_probs,)

    return outputs  # output, present, attention_probs

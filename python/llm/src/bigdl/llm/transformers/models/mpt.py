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
# https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/modeling_llama.py
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

import warnings
import torch
from einops import rearrange
import math
import torch.nn.functional as F
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import create_kv_cache, append_kv_cache


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., :x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2:]
#     return torch.cat((-x2, x1), dim=-1)
#
#
# def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
#     # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
#     cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
#     sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
#     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed
#
#
# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states
#     go from (batch, num_key_value_heads, seqlen, head_dim) to
#     (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
#                                                            n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def mpt_multihead_attention_forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
    qkv = self.Wqkv(x)
    if self.clip_qkv:
        qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
    (query, key, value) = qkv.chunk(3, dim=2)
    key_padding_mask = attention_mask
    if self.qk_ln:
        dtype = query.dtype
        query = self.q_ln(query).to(dtype)
        key = self.k_ln(key).to(dtype)
    (context, attn_weights, past_key_value) = mpt_scaled_multihead_dot_product_attention(query, key, value, self.n_heads,
                                                           past_key_value=past_key_value,
                                                           softmax_scale=self.softmax_scale, attn_bias=attn_bias,
                                                           key_padding_mask=key_padding_mask, is_causal=is_causal,
                                                           dropout_p=self.attn_dropout_p, training=self.training,
                                                           needs_weights=needs_weights)
    return (self.out_proj(context), attn_weights, past_key_value)


def mpt_scaled_multihead_dot_product_attention(query, key, value, n_heads,
                                               past_key_value=None,
                                               softmax_scale=None,
                                               attn_bias=None,
                                               key_padding_mask=None,
                                               is_causal=False,
                                               dropout_p=0.0,
                                               training=False,
                                               needs_weights=False,
                                               multiquery=False):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    bsz, n_heads, q_len, head_dim = q.size()
    device = q.device
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
    kv_seq_len = k.shape[-1]
    if past_key_value is not None:
        if len(past_key_value) != 0:
        #     k = torch.cat([past_key_value[0], k], dim=3)
        #     v = torch.cat([past_key_value[1], v], dim=2)
            cache_k = past_key_value[0].permute(0, 1, 3, 2)
            cache_v = past_key_value[1]
            kv_seq_len += cache_k.shape[-2]
            if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
                if device.type == 'xpu':
                    torch.xpu.empty_cache()
                # allocate new
                new_cache_k, new_cache_v = create_kv_cache(bsz,
                                                           kv_n_heads,  # Support GQA
                                                           head_dim,
                                                           cache_k.size(2),
                                                           kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                           dtype=cache_k.dtype,
                                                           device=device)
                new_cache_k[:] = cache_k
                new_cache_v[:] = cache_v
                cache_k = new_cache_k
                cache_v = new_cache_v
            key_states, value_states = append_kv_cache(cache_k, cache_v, k.permute(0, 1, 3, 2), v)
            k = key_states.permute(0, 1, 3, 2)
            v = value_states
        else:
            max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_key_states, new_value_states = create_kv_cache(bsz,
                                                               kv_n_heads,
                                                               head_dim,
                                                               kv_seq_len,
                                                               max_cache_length,
                                                               dtype=k.dtype,
                                                               device=device)
            new_key_states[:] = k.permute(0, 1, 3, 2)
            new_value_states[:] = v
            k = new_key_states
            v = new_value_states
        past_key_value = (k, v)
    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return (out, attn_weight, past_key_value)
    return (out, None, past_key_value)

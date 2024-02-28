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
# https://huggingface.co/mosaicml/mpt-7b-chat/blob/main/attention.py
#

import warnings
import torch
from einops import rearrange
import math
import torch.nn.functional as F
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import extend_kv_cache, init_kv_cache, append_kv_cache


KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def mpt_multihead_attention_forward(self, x, past_key_value=None, attn_bias=None,
                                    attention_mask=None, is_causal=True,
                                    needs_weights=False, rotary_emb_w_meta_info=None,
                                    **kwargs):
    qkv = self.Wqkv(x)
    if self.clip_qkv:
        qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
    (query, key, value) = qkv.chunk(3, dim=2)
    key_padding_mask = attention_mask
    if self.qk_ln:
        dtype = query.dtype
        query = self.q_ln(query).to(dtype)
        key = self.k_ln(key).to(dtype)

    if rotary_emb_w_meta_info is not None:
            rotary_emb = rotary_emb_w_meta_info['rotary_emb']
            seq_len = rotary_emb_w_meta_info['seq_len']
            offset_info = rotary_emb_w_meta_info['offset_info']
            bsz, seqlen = query.shape[:2]
            query = query.view(bsz, seqlen, -1, self.head_dim)
            key = key.view(bsz, seqlen, -1, self.head_dim)

            if rotary_emb_w_meta_info['impl'] == 'dail':
                value = value.view(bsz, seqlen, -1, self.head_dim)

                kv = torch.stack([key, value], dim=2)
                query, kv = rotary_emb(query,
                                       kv,
                                       seqlen_offset=offset_info,
                                       max_seqlen=seq_len)
                [key, value] = torch.unbind(kv, dim=2)

                value = value.view(bsz, seqlen, self.kv_n_heads * self.head_dim)
            elif rotary_emb_w_meta_info['impl'] == 'hf':
                (cos, sin) = rotary_emb(value, seq_len)
                if is_transformers_version_gte('4.36'):
                    query, key = apply_rotary_pos_emb(query,
                                                      key,
                                                      cos,
                                                      sin,
                                                      offset_info,
                                                      unsqueeze_dim=2)
                else:
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    query, key = apply_rotary_pos_emb(query, key, cos, sin,
                                                      offset_info)
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)

    (context, attn_weights, past_key_value) = \
        mpt_scaled_multihead_dot_product_attention(query, key, value, self.n_heads,
                                                   past_key_value=past_key_value,
                                                   softmax_scale=self.softmax_scale,
                                                   attn_bias=attn_bias,
                                                   key_padding_mask=key_padding_mask,
                                                   is_causal=is_causal,
                                                   dropout_p=self.attn_dropout_p,
                                                   training=self.training,
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
            # k = torch.cat([past_key_value[0], k], dim=3)
            # v = torch.cat([past_key_value[1], v], dim=2)
            cache_k = past_key_value[0].transpose(2, 3)
            cache_v = past_key_value[1]
            kv_seq_len += cache_k.shape[-2]
            if cache_k.stride()[1] < kv_seq_len * cache_k.size(3):
                # allocate new
                new_cache_k, new_cache_v = extend_kv_cache(bsz,
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
            key_states, value_states = append_kv_cache(cache_k, cache_v, k.transpose(2, 3), v)
            k = key_states.transpose(2, 3)
            v = value_states
        else:
            max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_key_states, new_value_states = init_kv_cache(bsz,
                                                             kv_n_heads,
                                                             head_dim,
                                                             kv_seq_len,
                                                             max_cache_length,
                                                             dtype=k.dtype,
                                                             device=device)
            new_key_states[:] = k.transpose(2, 3)
            new_value_states[:] = v
            k = new_key_states.transpose(2, 3)
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
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k \
                or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            invalidInputError(False, f'attn_bias (shape: {attn_bias.shape}) '
                                     f'is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propogating key_padding_mask to the attention module '
                          + 'and applying it within the attention module can cause '
                          + 'unneccessary computation/memory usage. Consider integrating '
                          + 'into attn_bias once and passing that to each attention '
                          + 'module instead.')
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
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p,
                                                  training=training, inplace=True)
    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return (out, attn_weight, past_key_value)
    return (out, None, past_key_value)

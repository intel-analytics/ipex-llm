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
# https://huggingface.co/Qwen/Qwen-7B-Chat/blob/be72f02dd47087f9035ee9bb5dea571b84785d27/modeling_qwen.py
#
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import math
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.utils import logging

try:
    from einops import rearrange
except ImportError:
    rearrange = None

from bigdl.llm.transformers.models.utils import extend_kv_cache, init_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import rotate_half
from bigdl.llm.utils.common import invalidInputError, invalidOperationError
from bigdl.llm.ggml.quantize import ggml_tensor_qtype

apply_rotary_emb_func = None

flash_attn_unpadded_func = None

logger = logging.get_logger(__name__)

KV_CACHE_ALLOC_BLOCK_LENGTH = 256
SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2


def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = (t_ * cos) + (rotate_half(t_) * sin)
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


def qwen_attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb_list: Optional[List[torch.Tensor]] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):
    invalidInputError(not self.use_flash_attn and not self.use_cache_quantization,
                      "flash attn and kv_cache quantization are not supported")

    mixed_x_layer = self.c_attn(hidden_states)
    query, key, value = mixed_x_layer.split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    kv_seq_len = hidden_states.size()[1]

    if rotary_pos_emb_list is not None:
        cur_len = query.shape[1]
        if len(rotary_pos_emb_list) == 1:
            if query.device.type == 'xpu':
                cos, sin = rotary_pos_emb_list[0]
                cos = cos[:, -cur_len:, :, :]
                sin = sin[:, -cur_len:, :, :]
                rot_dim = cos.shape[-1]
                query_cur = query[..., :rot_dim]
                key_cur = key[..., :rot_dim]
                torch.ops.torch_ipex.apply_rotary_embedding(query_cur, sin, cos, query_cur)
                torch.ops.torch_ipex.apply_rotary_embedding(key_cur, sin, cos, key_cur)
            else:
                rotary_pos_emb = rotary_pos_emb_list[0]
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # Slice the pos emb for current inference
                query = apply_rotary_pos_emb(query, q_pos_emb)
                key = apply_rotary_pos_emb(key, k_pos_emb)
        else:
            query_list = []
            key_list = []
            for i, rotary_pos_emb in enumerate(rotary_pos_emb_list):
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # Slice the pos emb for current inference
                query_list += [apply_rotary_pos_emb(query[i:i+1, :, :], q_pos_emb)]
                key_list += [apply_rotary_pos_emb(key[i:i+1, :, :], k_pos_emb)]
            query = torch.cat(query_list, dim=0)
            key = torch.cat(key_list, dim=0)

    bsz, _, n_heads, head_dim = key.size()

    if layer_past is not None:
        cache_k, cache_v = layer_past[0], layer_past[1]
        cache_k = cache_k.transpose(1, 2)
        cache_v = cache_v.transpose(1, 2)
        kv_seq_len += cache_k.shape[2]
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=hidden_states.device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v

        key_states, value_states = append_kv_cache(cache_k, cache_v,
                                                   key.transpose(1, 2), value.transpose(1, 2))
        key = key_states
        value = value_states
    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        new_key_states, new_value_states = init_kv_cache(bsz,
                                                         self.num_heads,
                                                         self.head_dim,
                                                         kv_seq_len,
                                                         max_cache_length,
                                                         dtype=key.dtype,
                                                         device=hidden_states.device)
        new_key_states[:] = key.transpose(1, 2)
        new_value_states[:] = value.transpose(1, 2)
        key = new_key_states
        value = new_value_states

    query_size, key_size = query.size(1), key.size(2)
    if key_size > self.seq_length and self.use_logn_attn and not self.training:
        seq_start = key_size - query_size
        seq_end = key_size
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
        query = query * logn_tensor.expand_as(query)
    if query_size == key_size:
        causal_mask = torch.tril(
            torch.ones((key_size, key_size), dtype=torch.bool, device=query.device)
        ).view(1, 1, key_size, key_size)
    else:
        causal_mask = None
    query = query.transpose(1, 2)

    attn_output, attn_weight = self._attn(
        query, key, value, causal_mask, attention_mask, head_mask
    )
    context_layer = self._merge_heads(
        attn_output, self.num_heads, self.head_dim
    )

    attn_output = self.c_proj(context_layer)

    if use_cache:
        outputs = (attn_output, (key.transpose(1, 2), value.transpose(1, 2)))
    else:
        outputs = (attn_output, None)
    if output_attentions:
        outputs += (attn_weight,)

    return outputs


def qwen_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    if x_2d.shape[0] == 1 and x.device.type == 'xpu' \
            and self.w2.qtype == ggml_tensor_qtype["sym_int4"] \
            and not (self.training and x.requires_grad):
        import linear_q4_0
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        return self.c_proj(linear_q4_0.mlp_forward_q4_0_xpu(
            x_2d, self.w2.weight.data, self.w1.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.w2.out_len,
        ))
    return self.c_proj(F.silu(self.w2(x)) * self.w1(x))

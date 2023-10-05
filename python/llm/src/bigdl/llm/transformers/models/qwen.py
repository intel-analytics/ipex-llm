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
# https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/modeling_qwen.py
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
from bigdl.llm.utils.common import invalidInputError

apply_rotary_emb_func = None

flash_attn_unpadded_func = None

logger = logging.get_logger(__name__)

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    if apply_rotary_emb_func is not None:
        t_ = t.float()
        freqs = freqs.squeeze(0).squeeze(1)
        cos = freqs[:, : freqs.shape[-1] // 2].cos()
        sin = freqs[:, : freqs.shape[-1] // 2].sin()
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs.shape[-1]
        t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass_ = t_pass_.float()
        t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
        return torch.cat((t_, t_pass_), dim=-1).type_as(t)


def qwen_attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):
    mixed_x_layer = self.c_attn(hidden_states)
    query, key, value = mixed_x_layer.split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    kv_seq_len = hidden_states.size()[1]

    if layer_past:
        # layer past[0] shape: bs * seq_len * head_num * dim
        kv_seq_len += layer_past[0].shape[1]
    if (
        self.use_dynamic_ntk
        and kv_seq_len == hidden_states.size()[1]
        and not self.training
    ):
        context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        self._ntk_cached = ntk_alpha
    else:
        ntk_alpha = self._ntk_cached
    rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha).to(
        hidden_states.device
    )

    if rotary_pos_emb is not None:
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = (rotary_pos_emb,) * 2

    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        # Slice the pos emb for current inference
        cur_len = query.shape[1]
        q_pos_emb = q_pos_emb[:, -cur_len:, :, :]
        k_pos_emb = k_pos_emb[:, -cur_len:, :, :]
        query = apply_rotary_pos_emb(query, q_pos_emb)
        key = apply_rotary_pos_emb(key, k_pos_emb)

    bsz, _, n_heads, head_dim = key.size()

    if layer_past is not None:
        # past_key, past_value = layer_past[0], layer_past[1]
        # key = torch.cat((past_key, key), dim=1)
        # value = torch.cat((past_value, value), dim=1)
        cache_k = layer_past[0].transpose(1, 2)
        cache_v = layer_past[1].transpose(1, 2)
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,  # Support GQA
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
        key = key_states.transpose(1, 2)
        value = value_states.transpose(1, 2)
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
        key = new_key_states.transpose(1, 2)
        value = new_value_states.transpose(1, 2)

    if use_cache:
        present = (key, value)
    else:
        present = None

    if self.use_logn_attn and not self.training:
        if self.logn_tensor.device != query.device or self.logn_tensor.dtype != query.dtype:
            self.logn_tensor = self.logn_tensor.to(query.device).type_as(query)
        seq_start = key.size(1) - query.size(1)
        seq_end = key.size(1)
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
        query = query * logn_tensor.expand_as(query)

    if (
        self.use_flash_attn
        and flash_attn_unpadded_func is not None
        and not self.is_fp32
        and query.is_cuda
    ):
        q, k, v = query, key, value
        context_layer = self.core_attention_flash(q, k, v)

        context_layer = rearrange(
            context_layer, "b s h d -> b s (h d)"
        ).contiguous()
    else:
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attn_output, attn_weight = self._attn(
            query, key, value, attention_mask, head_mask
        )
        context_layer = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )

    attn_output = self.c_proj(context_layer)
    outputs = (attn_output, present)
    if output_attentions:
        if (
            self.use_flash_attn
            and flash_attn_unpadded_func is not None
            and not self.is_fp32
        ):
            invalidInputError("Cannot output attentions while using flash-attn")
        else:
            outputs += (attn_weight,)

    return outputs

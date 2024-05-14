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
# https://huggingface.co/Qwen/Qwen-VL-Chat/blob/bbe5a805de49a41b7343d240ab84d4c305caa265/modeling_qwen.py
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
from ipex_llm.transformers.models.utils import extend_kv_cache, init_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import rotate_half
from ipex_llm.transformers.models.utils import use_sdp
from ipex_llm.transformers.models.utils import use_decoding_fast_path

import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    rot_dim = freqs[0].shape[-1]
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = (t_ * cos) + (rotate_half(t_) * sin)
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        # Due to issue https://github.com/intel/intel-extension-for-pytorch/issues/454,
        # currently put interpolation execution into cpu
        return F.interpolate(
            abs_pos.to("cpu").float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype).to(abs_pos.device)
    else:
        return abs_pos


def qwen_attention_forward_vl(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb: Optional[List[torch.Tensor]] = None,
    registered_causal_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):

    kv_seq_len = hidden_states.size()[1]

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    use_fuse_rope = should_use_fuse_rope(self, hidden_states)
    decoding_fast_path = use_decoding_fast_path(self.q_proj,
                                                use_fuse_rope,
                                                True,
                                                bsz * q_len)
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k, cache_v = layer_past[0], layer_past[1]
        cache_k = cache_k.transpose(1, 2)
        cache_v = cache_v.transpose(1, 2)

        kv_seq_len = cache_k.shape[-2]
        self.position_ids = self.position_ids.to(device)
        position_ids = self.position_ids[kv_seq_len]
        base = self.rope_base
        if is_enough_kv_cache_room(layer_past, kv_seq_len):
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

        args = [hidden_states, self.q_proj.weight.data, self.k_proj.weight.data,
                self.v_proj.weight.data, self.q_proj.bias.data, self.k_proj.bias.data,
                self.v_proj.bias.data, position_ids, cache_k, cache_v, self.q_proj.weight.qtype,
                self.v_proj.weight.qtype, kv_seq_len, self.head_dim, base]
        import linear_q4_0
        query, key, value = linear_q4_0.forward_qkv_bias(*args)
        kv_seq_len += 1
        query_size, key_size = 1, 1
    else:
        query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        # TODO: speed up
        # mixed_x_layer = self.c_attn(hidden_states)
        # query, key, value = mixed_x_layer.split(self.split_size, dim=2)

        # query = self._split_heads(query, self.num_heads, self.head_dim)
        # key = self._split_heads(key, self.num_heads, self.head_dim)
        # value = self._split_heads(value, self.num_heads, self.head_dim)
        if rotary_pos_emb is not None:
            cur_len = query.shape[1]
            rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
            rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # Slice the pos emb for current inference
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)
        query_size, key_size = query.size(1), key.size(1)

    if layer_past is not None:
        if not decoding_fast_path:
            kv_seq_len += layer_past[0].shape[1]
            # past_key, past_value = layer_past[0], layer_past[1]
            # key = torch.cat((past_key, key), dim=1)
            # value = torch.cat((past_value, value), dim=1)
            cache_k = layer_past[0].transpose(1, 2)
            cache_v = layer_past[1].transpose(1, 2)
            if cache_k.stride()[1] < kv_seq_len * cache_k.size(3):
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

    if use_cache:
        present = (key.transpose(1, 2), value.transpose(1, 2))
    else:
        present = None

    if decoding_fast_path:
        # change to (bsz, q_len, num_heads, head_dim)
        query = query.transpose(1, 2)

    if self.use_logn_attn and not self.training:
        if self.logn_tensor.device != query.device or self.logn_tensor.dtype != query.dtype:
            self.logn_tensor = self.logn_tensor.to(query.device).type_as(query)
        seq_start = key_size - key_size
        seq_end = key_size
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
        query = query * logn_tensor.expand_as(query)

    query = query.permute(0, 2, 1, 3)

    if not self.training and not hidden_states.requires_grad and \
            use_sdp(q_len, key.shape[2], self.head_dim, query):
        import linear_q4_0
        attn_output = linear_q4_0.sdp(query, key, value, attention_mask)
        attn_output = attn_output.view(query.shape)
        attn_output = attn_output.transpose(1, 2)
        attn_weight = None
    else:
        attn_output, attn_weight = self._attn(
            query, key, value, registered_causal_mask, attention_mask, head_mask
        )

    context_layer = self._merge_heads(
        attn_output, self.num_heads, self.head_dim
    )

    attn_output = self.c_proj(context_layer)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weight,)

    return outputs


def should_use_fuse_rope(self, query_states):
    use_fuse_rope = query_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and query_states.requires_grad)
    return use_fuse_rope


def is_enough_kv_cache_room(layer_past, kv_seq_len=1):
    # to determinate if is enough kv cache room in transformers between 4.31 and 4.35
    # seq_len for current seq len
    # For llama like kv cache, i.e., [bs, n_head, seq_len, head_dim]
    if layer_past is None:
        return False
    else:
        cache_k, cache_v = layer_past[0], layer_past[1]
        cache_k = cache_k.transpose(1, 2)
        return cache_k.stride(1) < (kv_seq_len + 1) * cache_k.size(3)


def qwen_vl_resampler_forward(self, x, attn_mask=None):

    pos_embed = get_abs_pos(self.pos_embed, x.size(1))

    x = self.kv_proj(x)
    x = self.ln_kv(x).permute(1, 0, 2)

    N = x.shape[1]
    q = self.ln_q(self.query)
    out = self.attn(
        self._repeat(q, N) + self.pos_embed.unsqueeze(1),
        x + pos_embed.unsqueeze(1),
        x,
        attn_mask=attn_mask)[0]
    return out.permute(1, 0, 2)


def qwen_vl_vision_transformer_forward(self, x: torch.Tensor):
    x = x.to(
        dtype=self.transformer.get_cast_dtype(),
        device=self.transformer.get_cast_device(),
    )
    # to patches
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    x = x + get_abs_pos(self.positional_embedding, x.size(1))

    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.attn_pool(x)
    x = self.ln_post(x)
    x = x @ self.proj

    return x

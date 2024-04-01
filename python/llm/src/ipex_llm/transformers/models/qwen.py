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

from ipex_llm.transformers.models.utils import extend_kv_cache, init_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache
from ipex_llm.transformers.models.utils import rotate_half, SILU
from ipex_llm.transformers.models.utils import mlp_fusion_check
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb_cache_freq_xpu
from ipex_llm.transformers.models.utils import use_flash_attention, use_esimd_sdp
from ipex_llm.transformers.models.utils import decoding_fast_path_qtype_check
from ipex_llm.utils.common import invalidInputError, invalidOperationError
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from transformers.modeling_outputs import BaseModelOutputWithPast

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
        cache_v = cache_v.transpose(1, 2)
        return cache_k.stride(1) < (kv_seq_len + 1) * cache_k.size(3)


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
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache(self.q_proj, hidden_states):
        forward_function = qwen_attention_forward_quantized
    else:
        forward_function = qwen_attention_forward_original
    return forward_function(
        self,
        hidden_states,
        rotary_pos_emb_list,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        output_attentions,
        use_cache,
    )


def qwen_attention_forward_original(
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
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype
    position_ids = rotary_pos_emb_list[-1] # the last one is posisiton_ids
    rotary_pos_emb_list = rotary_pos_emb_list[:-1]

    use_fuse_rope = should_use_fuse_rope(self, hidden_states)
    qtype_check = decoding_fast_path_qtype_check(self.q_proj)
    decoding_fast_path = (qtype_check and use_fuse_rope and bsz * q_len == 1)
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k, cache_v = layer_past[0], layer_past[1]
        cache_k = cache_k.transpose(1, 2)
        cache_v = cache_v.transpose(1, 2)

        kv_seq_len = cache_k.shape[-2]
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
        if len(rotary_pos_emb_list) != 0:
            cur_len = query.shape[1]
            if len(rotary_pos_emb_list) == 1:
                rotary_pos_emb = rotary_pos_emb_list[0]
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                if use_fuse_rope:
                    cos, sin = rotary_pos_emb
                    cos = cos.to(query.dtype)
                    sin = sin.to(query.dtype)
                    query, key = apply_rotary_pos_emb_cache_freq_xpu(query, key, sin, cos, "qwen")
                else:
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
                    if use_fuse_rope:
                        cos, sin = rotary_pos_emb
                        cos = cos.to(query.dtype)
                        sin = sin.to(query.dtype)
                        query, key = apply_rotary_pos_emb_cache_freq_xpu(query, key,
                                                                         sin, cos, "qwen")
                        query_list += [query]
                        key_list += [key]
                    else:
                        rotary_pos_emb = (rotary_pos_emb,) * 2
                        q_pos_emb, k_pos_emb = rotary_pos_emb
                        # Slice the pos emb for current inference
                        query_list += [apply_rotary_pos_emb(query[i:i+1, :, :], q_pos_emb)]
                        key_list += [apply_rotary_pos_emb(key[i:i+1, :, :], k_pos_emb)]
                query = torch.cat(query_list, dim=0)
                key = torch.cat(key_list, dim=0)
        query_size, key_size = query.size(1), key.size(1)
        kv_seq_len = key_size if layer_past is None else key_size + layer_past[0].size(1)

    if kv_seq_len > self.seq_length and self.use_logn_attn and not self.training:
        seq_start = kv_seq_len - query_size
        seq_end = kv_seq_len
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
        query = query * logn_tensor.expand_as(query)

    if query_size > 1:
        causal_mask = torch.tril(
            torch.ones((kv_seq_len, kv_seq_len), dtype=torch.bool, device=query.device)
        ).view(1, 1, kv_seq_len, kv_seq_len)
        causal_mask = causal_mask[
            :, :, kv_seq_len - query_size:kv_seq_len, :kv_seq_len
        ]
    else:
        causal_mask = None

    if layer_past is not None:
        if not decoding_fast_path:
            cache_k, cache_v = layer_past[0], layer_past[1]
            cache_k = cache_k.transpose(1, 2)
            cache_v = cache_v.transpose(1, 2)
            if cache_k.stride(1) < kv_seq_len * cache_k.size(3):
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

    if not decoding_fast_path:
        query = query.transpose(1, 2)

    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query, key):
        attn_output = F.scaled_dot_product_attention(query.to(device, dtype=torch.float16),
                                                     key.to(device, dtype=torch.float16),
                                                     value.to(device, dtype=torch.float16),
                                                     is_causal=True)
        attn_output = attn_output.view(query.shape)
        attn_output = attn_output.transpose(1, 2)
        attn_weights = None
    elif not self.training and not hidden_states.requires_grad and \
            use_esimd_sdp(q_len, key.shape[2], self.head_dim, query):
        import linear_fp16_esimd
        attn_output = linear_fp16_esimd.sdp_forward(query,
                                                    key,
                                                    value)
        attn_output = attn_output.view(query.shape)
        attn_output = attn_output.transpose(1, 2)
        attn_weight = None
    else:
        attn_output, attn_weight = self._attn(
            query.to(key.dtype), key, value, causal_mask, attention_mask, head_mask
        )

    context_layer = self._merge_heads(
        attn_output, self.num_heads, self.head_dim
    )

    attn_output = self.c_proj(context_layer).to(original_dtype)

    if use_cache:
        outputs = (attn_output, (key.transpose(1, 2), value.transpose(1, 2)))
    else:
        outputs = (attn_output, None)
    if output_attentions:
        outputs += (attn_weight,)

    return outputs


def qwen_attention_forward_quantized(
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

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    position_ids = rotary_pos_emb_list[-1] # the last one is posisiton_ids
    rotary_pos_emb_list = rotary_pos_emb_list[:-1]

    use_fuse_rope = should_use_fuse_rope(self, hidden_states)
    # qtype_check = decoding_fast_path_qtype_check(self.q_proj)
    # TODO: use when decoding_fast_path = (qtype_check and use_fuse_rope and bsz * q_len == 1)
    decoding_fast_path = False
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        tmp_cache_k, tmp_cache_v = init_kv_cache(
            bsz,
            self.num_heads,
            self.head_dim,
            0,
            1,
            dtype=hidden_states.dtype,
            device=device
        )

        base = self.rope_base

        args = [hidden_states, self.q_proj.weight.data, self.k_proj.weight.data,
                self.v_proj.weight.data, self.q_proj.bias.data, self.k_proj.bias.data,
                self.v_proj.bias.data, position_ids, tmp_cache_k, tmp_cache_v,
                self.q_proj.weight.qtype, self.v_proj.weight.qtype, 0, self.head_dim, base]
        import linear_q4_0
        query, key, value = linear_q4_0.forward_qkv_bias(*args)
        self.kv_seq_len += 1
        kv_seq_len = self.kv_seq_len
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
        if rotary_pos_emb_list is not None:
            cur_len = query.shape[1]
            if len(rotary_pos_emb_list) == 1:
                rotary_pos_emb = rotary_pos_emb_list[0]
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                if use_fuse_rope:
                    cos, sin = rotary_pos_emb
                    cos = cos.to(query.dtype)
                    sin = sin.to(query.dtype)
                    query, key = apply_rotary_pos_emb_cache_freq_xpu(query, key, sin, cos, "qwen")
                else:
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
                    if use_fuse_rope:
                        cos, sin = rotary_pos_emb
                        cos = cos.to(query.dtype)
                        sin = sin.to(query.dtype)
                        query, key = apply_rotary_pos_emb_cache_freq_xpu(query, key,
                                                                         sin, cos, "qwen")
                        query_list += [query]
                        key_list += [key]
                    else:
                        rotary_pos_emb = (rotary_pos_emb,) * 2
                        q_pos_emb, k_pos_emb = rotary_pos_emb
                        # Slice the pos emb for current inference
                        query_list += [apply_rotary_pos_emb(query[i:i+1, :, :], q_pos_emb)]
                        key_list += [apply_rotary_pos_emb(key[i:i+1, :, :], k_pos_emb)]
                query = torch.cat(query_list, dim=0)
                key = torch.cat(key_list, dim=0)
        query_size, key_size = query.size(1), key.size(1)
        kv_seq_len = key_size if layer_past is None else key_size + layer_past[0].size(1)

    if kv_seq_len > self.seq_length and self.use_logn_attn and not self.training:
        seq_start = kv_seq_len - query_size
        seq_end = kv_seq_len
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
        query = query * logn_tensor.expand_as(query)

    if query_size > 1:
        causal_mask = torch.tril(
            torch.ones((kv_seq_len, kv_seq_len), dtype=torch.bool, device=query.device)
        ).view(1, 1, kv_seq_len, kv_seq_len)
        causal_mask = causal_mask[
            :, :, kv_seq_len - query_size:kv_seq_len, :kv_seq_len
        ]
    else:
        causal_mask = None

    if layer_past is None:
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        # query, key, value's shape: [bs, num_heads, seq_len, head_dim]

        # save kv seq len for decoding_fast_path
        self.kv_seq_len = key.shape[-2]
        # For first token, use original attn
        attn_output, attn_weight = self._attn(
            query, key, value, causal_mask, attention_mask, head_mask
        )
        if use_cache:
            max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
            k_cache, v_cache = init_fp8_kv_cache(
                query.size(0), self.num_heads, kv_seq_len, self.head_dim,
                device=query.device, new_layout=True
            )
            key, value = append_fp8_kv_cache(k_cache, v_cache, key, value)
    else:
        if decoding_fast_path:
            k_cache, v_cache = layer_past[0], layer_past[1]
            # k_cache and v_cache's shape: [bs, num_heads, context_length, head_dim]
        else:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            k_cache, v_cache = layer_past[0], layer_past[1]

        k_cache = k_cache.transpose(1, 2)
        v_cache = v_cache.transpose(1, 2)
        # k_cache and v_cache's shape: [bs, num_heads, context_length, head_dim]

        key, value = append_fp8_kv_cache(k_cache, v_cache, key, value, new_layout=True)

        attn_output, attn_weight = core_attn(
            self, query, key, value, causal_mask, attention_mask, head_mask
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


def core_attn(self, query, key, value, causal_mask=None, attention_mask=None, head_mask=None):
    if query.size(2) != 1 or query.device.type != 'xpu':
        # We have no CPU fp8 matmul implementation for now, so just upscale to fp32
        key, value = restore_fp8_kv_cache(key, value, query.dtype)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            if self.use_cache_quantization:
                size_temp = value[0].size(-1)
            else:
                size_temp = value.size(-1)
            attn_weights = attn_weights / (size_temp ** 0.5)

        mask_value = torch.finfo(attn_weights.dtype).min
        if causal_mask is not None:
            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if self.softmax_in_fp32:
            attn_weights = torch.nn.functional.softmax(attn_weights.float(), dim=-1)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.type(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # We have no CPU fp8 matmul implementation for now, so just upscale to fp32
        attn_output = torch.matmul(attn_weights, value)
    else:
        import linear_q4_0
        attn_output = linear_q4_0.sdp_fp8(query, key, value,
                                          attention_mask)
        attn_weights = None
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def qwen_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    qtype = getattr(self.w1, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training) and not self.w1.enable_xetla:
        import linear_q4_0
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        return self.c_proj(linear_q4_0.mlp_forward_xpu(
            x_2d, self.w2.weight.data, self.w1.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.w2.out_len,
            SILU, qtype
        ))
    return self.c_proj(F.silu(self.w2(x)) * self.w1(x))


def qwen_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(
            False,
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        invalidInputError(False, "You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        if self.use_cache_quantization:
            past_length = past_key_values[0][0][0].size(2)
        else:
            past_length = past_key_values[0][0].size(1)
    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    if attention_mask is not None:
        if batch_size <= 0:
            invalidInputError(False, "batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    hidden_states = inputs_embeds

    kv_seq_len = hidden_states.size()[1]
    if past_key_values[0] is not None:
        # past key values[0][0] shape: bs * seq_len * head_num * dim
        if self.use_cache_quantization:
            kv_seq_len += past_key_values[0][0][0].shape[2]
        else:
            kv_seq_len += past_key_values[0][0].shape[1]

    if self.training or not self.use_dynamic_ntk:
        ntk_alpha_list = [1.0]
    elif kv_seq_len != hidden_states.size()[1]:
        ntk_alpha_list = self.rotary_emb._ntk_alpha_cached_list
    else:
        ntk_alpha_list = []
        if attention_mask is not None and kv_seq_len > self.seq_length:
            true_seq_lens = attention_mask.squeeze(1).squeeze(1).eq(0).sum(dim=-1,
                                                                           dtype=torch.int32)
            for i in range(hidden_states.size()[0]):
                true_seq_len = true_seq_lens[i].item()
                ntk_alpha = self.get_ntk_alpha(true_seq_len)
                ntk_alpha_list.append(ntk_alpha)
        else:
            ntk_alpha = self.get_ntk_alpha(kv_seq_len)
            ntk_alpha_list.append(ntk_alpha)
    self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
    rotary_pos_emb_list = [
        self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
    ] + [position_ids]

    hidden_states = self.drop(hidden_states)
    output_shape = input_shape + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                rotary_pos_emb_list,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            # bigdl-llm changes
            curr_device = block.ln_1.weight.device
            from accelerate.utils.operations import send_to_device
            if rotary_pos_emb_list is not None:
                rotary_pos_emb_list = send_to_device(rotary_pos_emb_list, curr_device)
            if attention_mask is not None:
                attention_mask = send_to_device(attention_mask, curr_device)
            if head_mask[i] is not None:
                head_mask[i] = send_to_device(head_mask[i], curr_device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = send_to_device(encoder_hidden_states, curr_device)
            if encoder_attention_mask is not None:
                encoder_attention_mask = send_to_device(encoder_attention_mask,
                                                        curr_device)
            # bigdl-llm changes ends

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb_list=rotary_pos_emb_list,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, presents, all_hidden_states] if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )

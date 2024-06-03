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

import torch
import warnings
import importlib
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import math
import os
import torch.nn.functional as F
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import SILU
from ipex_llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache
from ipex_llm.transformers.models.utils import is_enough_kv_cache_room_4_31, \
    apply_rotary_pos_emb, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu
from ipex_llm.transformers.models.utils import use_flash_attention, use_sdp, use_sdp_fp8
from ipex_llm.transformers.models.utils import mlp_fusion_check, fp16_fusion_check
from ipex_llm.transformers.models.utils import use_decoding_fast_path
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel
from ipex_llm.transformers.low_bit_linear import SYM_INT4, FP8E5, IQ2_XXS, FP4
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.llama import should_use_fuse_rope, should_use_xetla_mm_qkv
from ipex_llm.transformers.models.llama import fuse_qkv_weight_xetla, repeat_kv, native_sdp
from ipex_llm.transformers.models.llama import llama_decoding_fast_path_qtype_check

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = Tuple[torch.Tensor]
from transformers import logging
KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def minicpm_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    forward_function = minicpm_attention_forward_original
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        kwargs=kwargs
    )


def minicpm_attention_forward_original(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(self.q_proj,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp

    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k = past_key_value.key_cache[self.layer_idx]
        cache_v = past_key_value.value_cache[self.layer_idx]
        kv_seq_len = cache_k.shape[-2]
        import xe_linear
        query_states, key_states, value_states = xe_linear.forward_qkv(hidden_states,
                                                                       self.q_proj.weight,
                                                                       self.k_proj.weight,
                                                                       self.v_proj.weight,
                                                                       position_ids,
                                                                       cache_k, cache_v,
                                                                       self.q_proj.weight.qtype,
                                                                       self.v_proj.weight.qtype,
                                                                       kv_seq_len,
                                                                       self.head_dim,
                                                                       self.rotary_emb.base,)
        kv_seq_len += 1
        # update past_key_value's seem_tokens and kv caches.
        if self.layer_idx == 0:
            past_key_value.seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states

    else:
        if self.config.pretraining_tp > 1:
            key_value_slicing = ((self.num_key_value_heads * self.head_dim) //
                                 self.config.pretraining_tp)
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim)
                                                    // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i])
                            for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i])
                          for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i])
                            for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            if fp16_fusion_check(self.q_proj, hidden_states, self.training) and \
                    hidden_size == 4096 and self.q_proj.out_features == self.k_proj.out_features:
                # only use mm_qkv_out on pvc for llama-7b
                if not hasattr(self, "qkv_proj_weight"):
                    self.qkv_proj_weight = torch.stack([self.q_proj.weight,
                                                        self.k_proj.weight,
                                                        self.v_proj.weight]).contiguous()
                    self.q_proj.weight.data = self.qkv_proj_weight[0, :, :]
                    self.k_proj.weight.data = self.qkv_proj_weight[1, :, :]
                    self.v_proj.weight.data = self.qkv_proj_weight[2, :, :]
                    torch.xpu.empty_cache()
                query_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                key_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                         dtype=hidden_states.dtype, device=hidden_states.device)
                value_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                torch.ops.torch_ipex.mm_qkv_out(
                    hidden_states, self.qkv_proj_weight, None,
                    query_states, key_states, value_states
                )
            else:
                if should_use_xetla_mm_qkv(self, device):
                    if not hasattr(self, "qkv_proj_qweight"):
                        self.qkv_proj_qweight = fuse_qkv_weight_xetla(self.q_proj,
                                                                      self.k_proj,
                                                                      self.v_proj,
                                                                      self.q_proj.weight.qtype,)
                    import xe_linear
                    q_out_len = self.q_proj.out_len
                    k_out_len = self.k_proj.out_len
                    v_out_len = self.v_proj.out_len
                    qkv_states = xe_linear.mm_xetla(hidden_states,
                                                    self.qkv_proj_qweight,
                                                    self.q_proj.weight.qtype)
                    query_states = qkv_states[:, :, :q_out_len]
                    key_states = qkv_states[:, :, q_out_len:q_out_len + k_out_len]
                    value_states = qkv_states[:, :, q_out_len + k_out_len:]
                else:
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                invalidInputError(False,
                                  "The cache structure has changed since version v4.36. "
                                  f"If you are using {self.__class__.__name__} for "
                                  "auto-regressive decodingwith k/v caching, please make sure "
                                  "to initialize the attention class with a layer index.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if use_fuse_rope:
            rope_theta = self.rotary_emb.base
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "llama",
                                                                         rope_theta=rope_theta)
        else:
            if cache_position is not None:
                # for transformers 4.38.0
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama2")
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama")

        if past_key_value is not None:
            # update the number of seen tokens
            if self.layer_idx == 0:
                past_key_value.seen_tokens += key_states.shape[-2]

            # reuse k, v, self_attention
            # update `past_key_value` with `key_states` and `value_states` for layer `layer_idx`
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(key_states)
                past_key_value.value_cache.append(value_states)
            else:
                cache_k = past_key_value.key_cache[self.layer_idx]
                cache_v = past_key_value.value_cache[self.layer_idx]

                if not enough_kv_room:
                    # allocate new
                    new_c_k, new_c_v = extend_kv_cache(bsz,
                                                       self.num_key_value_heads,  # Support GQA
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=device)

                    new_c_k[:] = cache_k
                    new_c_v[:] = cache_v
                    cache_k = new_c_k
                    cache_v = new_c_v

                key_states, value_states = append_kv_cache(cache_k,
                                                           cache_v,
                                                           key_states,
                                                           value_states)

                # update past_key_value
                past_key_value.key_cache[self.layer_idx] = key_states
                past_key_value.value_cache[self.layer_idx] = value_states

    if cache_position is not None:
        new_attention_mask = attention_mask[:, :, kv_seq_len - q_len:kv_seq_len, 0:kv_seq_len]
    else:
        new_attention_mask = attention_mask

    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, new_attention_mask):
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # now only use flash attention for first token
        attn_output = F.scaled_dot_product_attention(query_states.to(device, dtype=torch.float16),
                                                     key_states.to(device, dtype=torch.float16),
                                                     value_states.to(device, dtype=torch.float16),
                                                     is_causal=True)
        attn_weights = None
    elif not self.training and not hidden_states.requires_grad and \
            self.layer_idx > 0 and \
            use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import xe_addons
        attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                    new_attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # otherwise, use native attention
        if query_states.device.type == "xpu":
            attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                                   new_attention_mask, cache_position,
                                                   bsz, q_len, kv_seq_len,
                                                   self.head_dim, self.num_heads, output_attentions)
        else:
            # CPU path
            if not output_attentions:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=new_attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    # The q_len > 1 is necessary to match with
                    # AttentionMaskConverter.to_causal_4d that
                    # does not create a causal mask in case q_len == 1.
                    is_causal=self.is_causal and new_attention_mask is None and q_len > 1,
                )
            else:
                attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                                       new_attention_mask, cache_position,
                                                       bsz, q_len, kv_seq_len,
                                                       self.head_dim,
                                                       self.num_heads, output_attentions)

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value

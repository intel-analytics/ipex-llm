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
from ipex_llm.transformers.models.llama import should_split_qkv_tensor, should_split_qkv_tensor

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
    if use_quantize_kv_cache(self.q_proj, hidden_states):
        forward_function = minicpm_attention_forward_quantized
    else:
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


def minicpm_attention_forward_quantized(
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

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(self.q_proj,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        tmp_cache_k, tmp_cache_v = init_kv_cache(
            bsz,
            self.num_key_value_heads,
            self.head_dim,
            0,
            1,
            dtype=hidden_states.dtype,
            device=device
        )
        import xe_linear
        query_states, key_states, value_states = xe_linear.forward_qkv(hidden_states,
                                                                       self.q_proj.weight,
                                                                       self.k_proj.weight,
                                                                       self.v_proj.weight,
                                                                       position_ids,
                                                                       tmp_cache_k, tmp_cache_v,
                                                                       self.q_proj.weight.qtype,
                                                                       self.v_proj.weight.qtype,
                                                                       0,
                                                                       self.head_dim,
                                                                       self.rotary_emb.base,)
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
                invalidInputError(
                    False,
                    f"The cache structure has changed since version v4.36."
                    f" If you are using {self.__class__.__name__} "
                    f"for auto-regressive decoding with k/v caching,"
                    f" please make sure to initialize the attention class "
                    "with a layer index."
                )
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
    kv_seq_len = key_states.shape[-2]

    if len(past_key_value.key_cache) <= self.layer_idx:
        repeated_key_states = repeat_kv(key_states, self.num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, self.num_key_value_groups)
        if should_split_qkv_tensor(query_states, bsz, self.num_heads,
                                   q_len, kv_seq_len, output_attentions):
            attn_output, _ = native_sdp_split_qkv_tensor(query_states, repeated_key_states,
                                                         repeated_value_states,
                                                         attention_mask, cache_position,
                                                         bsz, q_len, kv_seq_len, self.head_dim,
                                                         self.num_heads)
        else:
            attn_weights = torch.matmul(query_states, repeated_key_states
                                        .transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size "
                    f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if cache_position is not None:
                    # for transformers 4.38.0
                    causal_mask = attention_mask[:, :, cache_position, : kv_seq_len]
                    attn_weights = attn_weights + causal_mask
                else:
                    attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                    if attention_mask.size() != attn_mask_size:
                        invalidInputError(False,
                                          f"Attention mask should be of size {attn_mask_size}, "
                                          f"but is {attention_mask.size()}")
                    attn_weights = attn_weights + attention_mask

            if kv_seq_len >= 2048 or bsz >= 64:
                # for memory considerations, do not upcast attention to fp32
                # for long sequences or large batches
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                     dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, repeated_value_states)
        if use_cache:
            cache_kwargs = None
            key_states, value_states = past_key_value.update(key_states, value_states,
                                                             self.layer_idx, cache_kwargs)
    else:
        cache_kwargs = None  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, cache_kwargs)
        kv_seq_len = key_states.shape[-2]
        if not use_sdp_fp8(q_len, key_states.shape[2], query_states):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            key_states = repeat_kv(key_states, self.num_key_value_groups)\
                .to(device, dtype=query_states.dtype)
            value_states = repeat_kv(value_states, self.num_key_value_groups)\
                .to(device, dtype=query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size"
                    f" {(bsz, self.num_heads, q_len, kv_seq_len)},"
                    f" but is {attn_weights.size()}"
                )

            if attention_mask is not None:
                if cache_position is not None:
                    # for transformers 4.38.0
                    causal_mask = attention_mask[:, :, cache_position, : kv_seq_len]
                    attn_weights = attn_weights + causal_mask
                else:
                    attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                    if attention_mask.size() != attn_mask_size:
                        invalidInputError(False,
                                          f"Attention mask should be of size {attn_mask_size}, "
                                          f"but is {attention_mask.size()}")
                    attn_weights = attn_weights + attention_mask

            if kv_seq_len >= 2048 or bsz >= 64:
                # for memory considerations, do not upcast attention to fp32
                # for long sequences or large batches
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                     dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import xe_addons
            if cache_position is not None:
                new_attn_mask = attention_mask[:, :, kv_seq_len-q_len:kv_seq_len, 0:kv_seq_len]
            else:
                new_attn_mask = attention_mask
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states, new_attn_mask)
            attn_weights = None

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(
            False,
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)},"
            f" but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size
                                                 // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i],
                                    o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def minicpm_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    from ipex_llm.transformers.kv import DynamicFp8Cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    input = input_ids if input_ids is not None else inputs_embeds
    if use_cache and use_quantize_kv_cache(self.layers[0].mlp.up_proj, input):
        if not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return minicpm_model_forward_internal(
        self=self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


def minicpm_model_forward_internal(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None \
        else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False,
                          "You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        invalidInputError(False,
                          "You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing."
                " Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask)\
            else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            # bigdl-llm changes:
            curr_device = decoder_layer.input_layernorm.weight.device
            if attention_mask is not None:
                attention_mask = attention_mask.to(curr_device)
            if position_ids is not None:
                position_ids = position_ids.to(curr_device)
            # bigdl-llm changes end
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache \
            else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                     if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

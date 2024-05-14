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
# https://github.com/huggingface/transformers/blob/v4.38.0/src/transformers/models/stablelm/modeling_stablelm.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import math
from typing import Optional, Tuple, List, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.stablelm.modeling_stablelm import StableLmAttention, StableLmModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.utils import extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb, \
    apply_rotary_pos_emb_cache_freq_xpu
from ipex_llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache
from ipex_llm.transformers.models.utils import is_enough_kv_cache_room_4_36
from ipex_llm.transformers.models.utils import use_flash_attention, use_sdp
from ipex_llm.transformers.models.mistral import should_use_fuse_rope, repeat_kv
try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = Tuple[torch.Tensor]

import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def merge_qkv(module: torch.nn.Module):
    if isinstance(module, StableLmAttention):
        new_weight = torch.cat([
            module.q_proj.weight.data,
            module.k_proj.weight.data,
            module.v_proj.weight.data,
        ], dim=0)

        if module.q_proj.bias is not None:
            qkv_proj = torch.nn.Linear(0, 0, bias=True)
            new_bias = torch.cat([
                module.q_proj.bias.data,
                module.k_proj.bias.data,
                module.v_proj.bias.data,
            ], dim=0)
            qkv_proj.bias = torch.nn.Parameter(new_bias, requires_grad=False)
        else:
            qkv_proj = torch.nn.Linear(0, 0, bias=False)
        qkv_proj.weight = torch.nn.Parameter(new_weight, requires_grad=False)
        qkv_proj.in_features = new_weight.size(1)
        qkv_proj.out_features = new_weight.size(0)
        module.qkv_proj = qkv_proj

        del module.q_proj, module.k_proj, module.v_proj


def stablelm_model_forward(
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
    if use_cache and use_quantize_kv_cache_stablelm(self.layers[0].self_attn.head_dim,
                                                    self.layers[0].mlp.up_proj,
                                                    input_ids):
        if not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return StableLmModel.forward(
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


def use_quantize_kv_cache_stablelm(head_dim: int, linear: torch.nn.Module, x: torch.Tensor) -> bool:
    return (head_dim == 64 or head_dim == 128) and use_quantize_kv_cache(linear, x)


def stablelm_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache_stablelm(self.head_dim, self.o_proj, hidden_states):
        forward_function = stablelm_attention_forward_quantized
    else:
        forward_function = stablelm_attention_forward_original
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )


def stablelm_attention_forward_original(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_value: Optional[Cache]=None,
        output_attentions: bool=False,
        use_cache: bool=False,
        **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx)

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    kv_seq_len = key_states.shape[-2]

    if past_key_value is not None:
        invalidInputError(self.layer_idx is not None,
                          "The cache structure has changed since version v4.36. "
                          f"If you are using {self.__class__.__name__} for "
                          "auto-regressive decodingwith k/v caching, please make sure "
                          "to initialize the attention class with a layer index.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim:],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim:],
    )
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    if use_fuse_rope:
        query_rot, key_rot = apply_rotary_pos_emb_cache_freq_xpu(query_rot,
                                                                 key_rot,
                                                                 sin,
                                                                 cos,
                                                                 "stablelm",
                                                                 position_ids)
    else:
        query_rot, key_rot = apply_rotary_pos_emb(query_rot,
                                                  key_rot,
                                                  cos,
                                                  sin,
                                                  position_ids,
                                                  "stablelm")

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

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

            key_states, value_states = append_kv_cache(cache_k, cache_v,
                                                       key_states, value_states)

            # update past_key_value
            past_key_value.key_cache[self.layer_idx] = key_states
            past_key_value.value_cache[self.layer_idx] = value_states

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, attention_mask):
        attn_output = F.scaled_dot_product_attention(query_states.to(device, dtype=torch.float16),
                                                     key_states.to(device, dtype=torch.float16),
                                                     value_states.to(device, dtype=torch.float16),
                                                     is_causal=True)
        attn_weights = None
    elif not self.training and not hidden_states.requires_grad and \
            use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import linear_q4_0
        attn_output = linear_q4_0.sdp(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        invalidInputError(
            attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len),
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)},"
            f" but is {attn_weights.size()}")

        if attention_mask is not None:
            invalidInputError(
                attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},"
                f" but is {attention_mask.size()}")

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = \
            nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        invalidInputError(
            attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim),
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)},"
            f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def stablelm_attention_forward_quantized(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_value: Optional[Cache]=None,
        output_attentions: bool=False,
        use_cache: bool=False,
        **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        invalidInputError(
            self.layer_idx is not None,
            f"The cache structure has changed since version v4.36. "
            "If you are using {self.__class__.__name__} "
            "for auto-regressive decoding with k/v caching, "
            "please make sure to initialize the attention class "
            "with a layer index."
        )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim:],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim:],
    )
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    if use_fuse_rope:
        query_rot, key_rot = apply_rotary_pos_emb_cache_freq_xpu(query_rot,
                                                                 key_rot,
                                                                 sin,
                                                                 cos,
                                                                 "stablelm",
                                                                 position_ids)
    else:
        query_rot, key_rot = apply_rotary_pos_emb(query_rot,
                                                  key_rot,
                                                  cos,
                                                  sin,
                                                  position_ids,
                                                  "stablelm")

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    kv_seq_len = key_states.shape[-2]
    if len(past_key_value.key_cache) <= self.layer_idx:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        invalidInputError(
            attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len),
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}"
            f", but is {attn_weights.size()}")

        if attention_mask is not None:
            invalidInputError(
                attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},"
                f" but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # at inference time, for memory considerations, may not need to upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        invalidInputError(
            attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim),
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}"
            f", but is {attn_output.size()}")
        if use_cache:
            cache_kwargs = None
            key_states, value_states = past_key_value.update(key_states, value_states,
                                                             self.layer_idx, cache_kwargs)
    else:
        cache_kwargs = None  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, cache_kwargs)
        kv_seq_len = key_states.shape[-2]
        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        else:
            import linear_q4_0
            attn_weights = linear_q4_0.query_key_fp8_matmul(query_states, key_states)

        attn_weights = attn_weights / math.sqrt(self.head_dim)

        invalidInputError(
            attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len),
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}"
            f", but is {attn_weights.size()}")

        if attention_mask is not None:
            invalidInputError(
                attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},"
                f" but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # at inference time, for memory considerations, may not need to upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import linear_q4_0
            attn_output = linear_q4_0.attn_value_fp8_matmul(attn_weights, value_states)

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    invalidInputError(attn_output.size() == attn_output_size,
                      f"`attn_output` should be of size {attn_output_size},"
                      f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value

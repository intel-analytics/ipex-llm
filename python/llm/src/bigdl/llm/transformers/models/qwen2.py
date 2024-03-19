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
# https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/qwen2/modeling_qwen2.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigdl.llm.transformers.models.llama import repeat_kv
from bigdl.llm.transformers.models.utils import extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import use_quantize_kv_cache, restore_fp8_kv_cache
from bigdl.llm.transformers.models.utils import is_enough_kv_cache_room_4_36
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_cache_freq_xpu
from bigdl.llm.transformers.kv import DynamicFp8Cache
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import use_flash_attention, use_esimd_sdp
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.qwen2.modeling_qwen2 import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = Tuple[torch.Tensor]
import logging
from transformers import logging


logger = logging.get_logger(__name__)

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def should_use_fuse_rope(self, query_states, position_ids):
    use_fuse_rope = query_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and query_states.requires_grad)
    use_fuse_rope = use_fuse_rope and position_ids is not None
    return use_fuse_rope


def qwen2_model_forward(
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
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if use_cache and use_quantize_kv_cache(self.layers[0].mlp.up_proj, input_ids):
        if not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return qwen2_model_forward_internal(
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


def qwen2_model_forward_internal(
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
    output_attentions = output_attentions if output_attentions is not None else \
        self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False,
                          "You cannot specify both decoder_input_ids and "
                          "decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        invalidInputError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`..."
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
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    flash_attn_2 = self._attn_implementation == "flash_attention_2"
    if attention_mask is not None and flash_attn_2 and use_cache:

        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            invalidInputError(
                False,
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2."
                " Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing "
                "the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and
                                            0 in attention_mask) else None
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

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
            # bigdl-llm changes
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
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else \
            next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache,
                                 all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache(self.q_proj, hidden_states):
        forward_function = qwen2_attention_forward_quantized
    elif hidden_states.device.type == "cpu":
        forward_function = qwen2_sdpa_attention_forward
    else:
        forward_function = qwen2_attention_forward_origin
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )


def qwen2_attention_forward_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[DynamicFp8Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    bsz, q_len, _ = hidden_states.size()

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
        invalidInputError(self.layer_idx is not None,
                          "The cache structure has changed since version v4.36. "
                          f"If you are using {self.__class__.__name__} "
                          "for auto-regressive decoding with k/v caching, "
                          "please make sure to initialize the attention class "
                          "with a layer index.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    if use_fuse_rope:
        query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(query_states, key_states,
                                                                       sin, cos, "qwen2",
                                                                       position_ids)
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, cache_kwargs,
                                                         new_layout=True)

    if q_len == 1 and query_states.device.type == 'xpu' and not self.training \
            and not hidden_states.requires_grad:
        import linear_q4_0
        attn_output = linear_q4_0.sdp_fp8(query_states, key_states, value_states,
                                          attention_mask)
        attn_weights = None
    else:
        key, value = restore_fp8_kv_cache(key_states, value_states, query_states.dtype)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        invalidInputError(attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len),
                          ("Attention weights should be of size "
                           f"{(bsz, self.num_heads, q_len, kv_seq_len)},"
                           "but is {attn_weights.size()}"))

        if attention_mask is not None:
            invalidInputError(attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                              (f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}"
                               f" but is {attention_mask.size()}"))

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                             training=self.training)

        attn_output = torch.matmul(attn_weights, value)

    invalidInputError(attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim),
                      "`attn_output` should be of size "
                      f"{(bsz, self.num_heads, q_len, self.head_dim)},"
                      f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
SYM_INT4 = ggml_tensor_qtype["sym_int4"]
FP8E5 = ggml_tensor_qtype["fp8_e5m2"]


def qwen2_attention_forward_origin(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx)
    qtype = getattr(self.q_proj, "qtype", None)
    qtype_check = qtype in [SYM_INT4, FP8E5]
    decoding_fast_path = (qtype_check and use_fuse_rope
                          and enough_kv_room and bsz * q_len == 1)
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k = past_key_value.key_cache[self.layer_idx]
        cache_v = past_key_value.value_cache[self.layer_idx]
        kv_seq_len = cache_k.shape[-2]
        import linear_q4_0
        args = [hidden_states, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                self.q_proj.bias, self.k_proj.bias, self.v_proj.bias, position_ids, cache_k,
                cache_v, self.q_proj.weight.qtype, self.v_proj.weight.qtype, kv_seq_len,
                self.head_dim, self.rotary_emb.base]
        query_states, key_states, value_states = linear_q4_0.forward_qkv_bias(*args)
        kv_seq_len += 1
        if self.layer_idx == 0:
            past_key_value.seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states

    else:

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = \
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = \
            value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                invalidInputError(
                    False,
                    "The cache structure has changed since version v4.36. "
                    f"If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, "
                    "please make sure to initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(query_states, key_states,
                                                                           sin, cos, "qwen2",
                                                                           position_ids)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids)

        if past_key_value is not None:
            # update the number of seen tokens
            if self.layer_idx == 0:
                past_key_value.seen_tokens += key_states.shape[-2]

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
            use_esimd_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import linear_fp16_esimd
        attn_output = linear_fp16_esimd.sdp_forward(query_states,
                                                    key_states,
                                                    value_states)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        invalidInputError(attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len),
                          ("Attention weights should be of size "
                           f"{(bsz, self.num_heads, q_len, kv_seq_len)},"
                           "but is {attn_weights.size()}"))

        if attention_mask is not None:
            invalidInputError(attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                              (f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}"
                               f" but is {attention_mask.size()}"))

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = \
            nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.attention_dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

    invalidInputError(attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim),
                      "`attn_output` should be of size "
                      f"{(bsz, self.num_heads, q_len, self.head_dim)},"
                      f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(hidden_states.dtype), attn_weights, past_key_value


def qwen2_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx)
    qtype = getattr(self.q_proj, "qtype", None)
    qtype_check = qtype in [SYM_INT4, FP8E5]
    decoding_fast_path = (qtype_check and use_fuse_rope
                          and enough_kv_room and bsz * q_len == 1)
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k = past_key_value.key_cache[self.layer_idx]
        cache_v = past_key_value.value_cache[self.layer_idx]
        kv_seq_len = cache_k.shape[-2]
        import linear_q4_0
        args = [hidden_states, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                self.q_proj.bias, self.k_proj.bias, self.v_proj.bias, position_ids, cache_k,
                cache_v, self.q_proj.weight.qtype, self.v_proj.weight.qtype, kv_seq_len,
                self.head_dim, self.rotary_emb.base]
        query_states, key_states, value_states = linear_q4_0.forward_qkv_bias(*args)
        kv_seq_len += 1
        if self.layer_idx == 0:
            past_key_value.seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states

    else:

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = \
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = \
            value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                invalidInputError(
                    False,
                    "The cache structure has changed since version v4.36. "
                    f"If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, "
                    "please make sure to initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(query_states, key_states,
                                                                           sin, cos, "qwen2",
                                                                           position_ids)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids)

        if past_key_value is not None:
            # update the number of seen tokens
            if self.layer_idx == 0:
                past_key_value.seen_tokens += key_states.shape[-2]

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

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    invalidInputError(attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len),
                      ("Attention weights should be of size "
                       f"{(bsz, self.num_heads, q_len, kv_seq_len)},"
                       "but is {attn_weights.size()}"))

    if attention_mask is not None:
        invalidInputError(attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                          (f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}"
                           f" but is {attention_mask.size()}"))

        attn_weights = attn_weights + attention_mask

    from torch.nn.functional import scaled_dot_product_attention as sdpa
    attn_output = sdpa(query_states,
                       key_states,
                       value_states,
                       attn_mask=attention_mask,
                       dropout_p=self.attention_dropout if self.training else 0.0,
                       is_causal=self.is_causal and attention_mask is None and q_len > 1)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

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
from typing import Optional, Tuple, Union, List

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

from ipex_llm.transformers.models.utils import should_use_fuse_rope
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, restore_fp8_kv_cache
from ipex_llm.transformers.models.utils import use_flash_attention, use_sdp, use_sdp_causal
from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicNormalCache
from ipex_llm.utils.common import invalidInputError

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.qwen2.modeling_qwen2 import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers import logging


logger = logging.get_logger(__name__)


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
    use_quantize_kv = (
        self.config.hidden_size != 3584     # disable quantize kv in specific model
        and use_quantize_kv_cache(self.layers[0].mlp.up_proj, input_ids)
    )
    if use_cache:
        if use_quantize_kv and not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
        if not use_quantize_kv and not isinstance(past_key_values, DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
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
        invalidInputError(False,
                          "You have to specify either decoder_input_ids or decoder_inputs_embeds")

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
            # ipex-llm changes
            curr_device = decoder_layer.input_layernorm.weight.device
            if attention_mask is not None:
                attention_mask = attention_mask.to(curr_device)
            if position_ids is not None:
                position_ids = position_ids.to(curr_device)
            # ipex-llm changes end
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


def merge_qkv(module: torch.nn.Module):
    if isinstance(module, Qwen2Attention):
        new_weight = torch.cat([
            module.q_proj.weight.data,
            module.k_proj.weight.data,
            module.v_proj.weight.data,
        ], dim=0)
        new_bias = torch.cat([
            module.q_proj.bias.data,
            module.k_proj.bias.data,
            module.v_proj.bias.data,
        ], dim=-1)

        qkv_proj = torch.nn.Linear(0, 0, bias=True)
        qkv_proj.weight = torch.nn.Parameter(new_weight, requires_grad=False)
        qkv_proj.bias = torch.nn.Parameter(new_bias, requires_grad=False)
        qkv_proj.in_features = new_weight.size(1)
        qkv_proj.out_features = new_weight.size(0)
        module.qkv_proj = qkv_proj

        del module.q_proj, module.k_proj, module.v_proj


def qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)

    attn_weights = None
    if query_states.device.type == "cpu":
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = sdpa(query_states,
                           key_states,
                           value_states,
                           attn_mask=attention_mask,
                           dropout_p=self.attention_dropout if self.training else 0.0,
                           is_causal=self.is_causal and attention_mask is None and q_len > 1)
    elif not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, attention_mask):
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = sdpa(query_states.to(device, dtype=torch.float16),
                           key_states.to(device, dtype=torch.float16),
                           value_states.to(device, dtype=torch.float16),
                           is_causal=True).to(hidden_states.dtype)
    elif use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import xe_addons
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                            attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                        attention_mask)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states,
                                               value_states, attention_mask)
    else:
        if isinstance(past_key_value, DynamicFp8Cache):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                   dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                                   training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

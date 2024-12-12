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
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
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


from typing import Optional, Tuple, List, Union

import math
import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from ipex_llm.utils.common.log4Error import invalidInputError
from ipex_llm.transformers.npu_models.common import merge_linear


def merge_qkv(module: torch.nn.Module):
    if isinstance(module, LlamaAttention):
        qkv_proj = merge_linear([
            module.q_proj,
            module.k_proj,
            module.v_proj,
        ])
        module.qkv_proj = qkv_proj
        del module.q_proj, module.k_proj, module.v_proj


def merge_mlp(module: torch.nn.Module):
    if isinstance(module, LlamaMLP):
        gate_up_proj = merge_linear([
            module.gate_proj,
            module.up_proj,
        ])
        module.gate_up_proj = gate_up_proj
        del module.gate_proj, module.up_proj


def llama_model_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        invalidInputError(False,
                          ("You cannot specify both input_ids and inputs_embeds at the same time, "
                           "and must specify either one"))

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0

    # ipex-llm changes start
    from ipex_llm.transformers.kv import DynamicNormalCache
    if use_cache and not isinstance(past_key_values, DynamicNormalCache):
        past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                                      device=inputs_embeds.device)
    # ipex-llm changes end

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                           cache_position, past_seen_tokens)

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
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
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

    # ipex-llm changes start
    next_cache = next_decoder_cache if use_cache else None
    # ipex-llm changes end
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache,
                                 all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    past_key_value = getattr(self, "past_key_value", past_key_value)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    else:
        causal_mask = None

    if query_states.size(2) == key_states.size(2):
        # first token
        from intel_npu_acceleration_library.functional import scaled_dot_product_attention
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            is_causal=self.is_causal and causal_mask is None and q_len > 1,
        )
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                   dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_mlp_forward(self, x):
    gate_up_proj = self.gate_up_proj(x)
    gate_proj, up_proj = gate_up_proj.chunk(2, dim=-1)
    down_proj = self.down_proj(self.act_fn(gate_proj) * up_proj)
    return down_proj

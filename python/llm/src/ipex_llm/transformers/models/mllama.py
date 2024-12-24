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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mllama.modeling_mllama import MllamaVisionAttention
from transformers.models.mllama.modeling_mllama import MllamaTextSelfAttention
from ipex_llm.transformers.models.utils import use_quantize_kv_cache
from ipex_llm.transformers.models.utils import should_use_fuse_rope
from ipex_llm.transformers.models.common import merge_qkv_base, attention_softmax
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.kv import DynamicNormalCache, DynamicFp8Cache
from ipex_llm.transformers.utils import invalidInputError


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, MllamaVisionAttention)
    merge_qkv_base(module, MllamaTextSelfAttention)


def mllama_vision_attention_forward(
    self,
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = None,
):
    bsz, q_len, _ = hidden_state.size()

    qkv = self.qkv_proj(hidden_state)
    qkv = qkv.view(bsz, q_len, 3 * self.num_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query, key, value = qkv.chunk(3, dim=1)

    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query, key.contiguous(), value.contiguous(),
        attention_softmax, False
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return output, attn_weights


def mllama_text_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cross_attention_states: Optional[torch.FloatTensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
    full_text_row_masked_out_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
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

    # IPEX-LLM OPT start: kv cache and quantize kv cache
    inputs = input_ids if input_ids is not None else inputs_embeds
    use_cache = True if inputs.device.type == "xpu" else use_cache
    use_quantize_kv = use_quantize_kv_cache(
        self.layers[0].mlp.down_proj, inputs,
        self.config.num_attention_heads // self.config.num_key_value_heads
    )
    if use_cache:
        if use_quantize_kv and not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
        elif not use_quantize_kv and not isinstance(past_key_values, DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    # IPEX-LLM OPT end

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    invalidInputError((input_ids is None) ^ (inputs_embeds is None),
                      "You cannot specify both input_ids and inputs_embeds at the same time, "
                      "and must specify either one")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    hidden_states = inputs_embeds

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # IPEX-LLM OPT start: use fused rope
    if (should_use_fuse_rope(hidden_states, position_ids, False)
            and self.rotary_emb.rope_type == "llama3"):
        position_embeddings = self.rotary_emb.inv_freq
    # IEPX_LLM OPT end

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # For text-only path we should skip cross attention layers.
        # Let's check if the layer is cross attention layer and if we have cross attention states
        # or cached cross attention states.
        is_cross_attention_layer = idx in self.cross_attention_layers

        # IPEX-LLM change start
        if is_cross_attention_layer and cross_attention_states is None:
            if past_key_values is None:
                # use_cache=False
                continue
            elif len(past_key_values.key_cache) <= idx:
                # first token but no cross_attention_states, means no image inputs
                past_key_values.key_cache.append([])
                past_key_values.value_cache.append([])
                continue
            elif past_key_values.key_cache[idx] == []:
                # next token but no cross kv cache, means no image inputs
                continue
        # IPEX-LLM change end

        layer_outputs = decoder_layer(
            hidden_states,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            attention_mask=causal_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                     if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def mllama_cross_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cross_attention_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    query_states = self.q_norm(query_states.view(-1, self.head_dim))
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if cross_attention_states is not None:
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = self.k_norm(key_states.view(-1, self.head_dim))
        key_states = key_states.view(bsz, -1, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        # if we have a new image + new tokens, we only computed key_states on that new image
        # we still update the cross key states, past_image, new_image. And use it!
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, None
        )
    else:
        key_states, value_states = (
            past_key_value.key_cache[self.layer_idx],
            past_key_value.value_cache[self.layer_idx],
        )

    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == key_states.size(2)
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

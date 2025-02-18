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

# This file is adapted from
# https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct/blob/main/modeling_baichuan.py


import math
import torch
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.utils import should_use_fuse_rope, repeat_kv
from ipex_llm.transformers.models.common import attention_softmax
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.kv import DynamicNormalCache


def pre_register_inv_freq(module: torch.nn.Module):
    if module.__class__.__name__ == "RotaryEmbedding":
        inv_freq = module.inv_freq
        del module.inv_freq
        module.register_buffer("inv_freq", inv_freq, persistent=False)


# copied from Baichuan M1
def custom_convolution(U, K):
    """
    U: Input matrix, shape (bs, seq, h, d)
    K: Convolution kernel, shape (w, h)
    Returns: Output matrix V, shape (bs, seq, h, d)
    """
    # h, w = K.shape
    w = K.size(-1)
    padding = (w - 1, 0)
    U_padded = F.pad(U, (0, 0, 0, 0, *padding))  # Shape becomes (bs, seq+w-1, h, d)
    U_unfolded = U_padded.unfold(1, w, 1)  # Shape becomes (bs, seq+w-1, h, d, w)
    V_unfolded = U_unfolded * K  # Shape remains (bs, seq, h, d, w)
    V = V_unfolded.sum(dim=-1)  # Shape becomes (bs, seq, h, d)
    return V


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    seqlens: Optional[torch.LongTensor] = None,
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
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    invalidInputError((input_ids is None) ^ (inputs_embeds is None),
                      "You cannot specify both input_ids and inputs_embeds at the same time, "
                      "and must specify either one")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    use_cache = True if inputs_embeds.device.type == "xpu" else use_cache

    # IPEX-LLM changes start: remove batch multi-pack and use ipex-llm's kv cache
    # kept for BC (non `Cache` `past_key_values` inputs)
    if use_cache and not isinstance(past_key_values, DynamicNormalCache):
        past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    # IPEX-LLM changes end

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

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    # position_embeddings = self.rotary_emb(hidden_states, position_ids)
    position_embeddings = None

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            seqlens=None,
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


def eager_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    seqlens: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
):
    invalidInputError(seqlens is None, "`seq_lens` must be None")

    bsz, q_len, _ = hidden_states.size()
    qkv = self.W_pack(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=2)
    # q, k, v: [bsz, seq_len, num_heads, head_dim]

    if past_key_value is None or past_key_value.get_seq_length(self.layer_idx) == 0:    # prefill
        self.last_k = key_states[:, -1:]
        self.last_v = value_states[:, -1:]

        key_states = custom_convolution(key_states, self.conv_k)
        value_states = custom_convolution(value_states, self.conv_v)
    else:
        new_key_states = (self.conv_k[0, 0, :, 0, :1] * self.last_k +
                          self.conv_k[0, 0, :, 0, 1:] * key_states)
        self.last_k = key_states
        key_states = new_key_states

        new_value_states = (self.conv_v[0, 0, :, 0, : 1] * self.last_v +
                            self.conv_v[0, 0, :, 0, 1:] * value_states)
        self.last_v = value_states
        value_states = new_value_states

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    # q, k, v: [bsz, num_heads, seq_len, head_dim]

    invalidInputError(should_use_fuse_rope(hidden_states, position_ids, self.training),
                      "fuse rope must be used")
    import xe_addons
    xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                   query_states, key_states)

    # ignore sliding window
    key_states, value_states = past_key_value.update(key_states, value_states,
                                                     self.layer_idx, None)
    if self.head_dim <= 128:
        attn_weights = None
        attn_output = scaled_dot_product_attention(
            query_states, key_states, value_states,
            attention_mask, q_len == key_states.size(2)
        )
    else:
        n_rep = self.num_heads // self.num_key_value_heads
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attention_softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

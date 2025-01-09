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
# https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/cb7fc748b78b7ea99772e4cf76db155729ce774e/modeling_baichuan.py
# and
# https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/c6f8592a60b4ad73c210b28dd2ab3cca51abbf93/modeling_baichuan.py

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, restore_fp8_kv_cache, \
    should_use_compresskv
from ipex_llm.transformers.models.utils import update_past_key_value
from ipex_llm.transformers.models.utils import should_use_fuse_rope
from ipex_llm.transformers.models.utils import use_sdp
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb
from ipex_llm.transformers.models.utils import is_enough_kv_cache_room_4_36
from ipex_llm.transformers.kv import DynamicCompressFp8Cache, DynamicCompressCache
import warnings
import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def pre_compute_inv_freq(module: torch.nn.Module):
    if module.__class__.__name__ == "RotaryEmbedding":
        inv_freq = module.inv_freq
        del module.inv_freq
        module.register_buffer("inv_freq", inv_freq, persistent=False)


def baichuan_model_7b_forward(
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
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # IPEX-LLM OPT: compress kv and quantize kv
    if use_cache:
        inputs = input_ids if input_ids is not None else inputs_embeds
        use_compress_kv = should_use_compresskv(inputs, inputs.shape[1])
        use_quantize_kv = use_quantize_kv_cache(self.layers[0].mlp.up_proj, inputs,
                                                self.config.num_attention_heads,
                                                self.config.num_attention_heads)
        if use_compress_kv and not isinstance(past_key_values,
                                              DynamicCompressCache):
            if use_quantize_kv:
                past_key_values = DynamicCompressFp8Cache.from_legacy_cache(past_key_values)
            else:
                past_key_values = DynamicCompressCache.from_legacy_cache(past_key_values)

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at \
                          the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        log4Error.invalidInputError("You have to specify either decoder_input_ids \
                                     or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        # IPEX-LLM OPT: compress kv
        if isinstance(past_key_values, DynamicCompressCache):
            past_key_values_length = past_key_values.get_seq_length()
        else:
            past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length,
                                    dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # IPEX-LLM OPT: compress kv
    use_compresskv = isinstance(past_key_values, DynamicCompressCache)

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # IPEX-LLM OPT: compress kv
        if not use_compresskv:
            past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            # IPEX-LLM OPT: compress kv
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values if use_compresskv else past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            # IPEX-LLM OPT: compress kv
            if use_compresskv:
                next_decoder_cache = past_key_values
            else:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

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


def baichuan_attention_forward_7b(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
):
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    # [CompressKV]
    use_compresskv = isinstance(past_key_value, DynamicCompressCache)

    qkv = self.W_pack(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    kv_seq_len = key_states.shape[2]
    if past_key_value is not None:
        # [CompressKV]
        if use_compresskv:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                           self.layer_idx)
        else:
            kv_seq_len += past_key_value[0].shape[2]

    # IPEX-LLM OPT: fuse rope
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids, "baichuan")
        query_states = query_states.to(hidden_states.dtype)
        key_states = key_states.to(hidden_states.dtype)

    # IPEX-LLM OPT: kv cache and quantize kv
    # [CompressKV]
    if use_compresskv:
        enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value,
                                                      self.layer_idx,
                                                      q_len)
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx,
            query_states, attention_mask, 1,
            self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH)
    else:
        use_quantize_kv = use_quantize_kv_cache(self.W_pack, hidden_states,
                                                self.num_heads, self.num_heads)
        key_states, value_states = update_past_key_value(
            past_key_value, key_states, value_states,
            kv_seq_len, use_quantize_kv, device
        )
        past_key_value = (key_states, value_states) if use_cache else None

    if self.training:
        warnings.warn("xops is not supported on Intel GPU, so just use normal implementation")

    # IPEX-LLM OPT: sdp
    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == kv_seq_len
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def baichuan_attention_forward_13b(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    qkv = self.W_pack(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    kv_seq_len = key_states.shape[2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    # IPEX-LLM OPT: kv cache and quantize kv
    use_quantize_kv = use_quantize_kv_cache(self.W_pack, hidden_states,
                                            self.num_heads, self.num_heads)
    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, use_quantize_kv, device
    )
    past_key_value = (key_states, value_states) if use_cache else None

    if self.training:
        warnings.warn("xops is not supported on Intel GPU, so just use normal implementation")

    if attention_mask is not None:
        if len(attention_mask.size()) == 4:
            attention_mask = attention_mask[:, :, -q_len:, :]
        else:
            attention_mask = attention_mask[None, :, -q_len:, :]

    if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                            attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                        attention_mask)
        attn_weights = None
    else:
        if use_quantize_kv:
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights.to(dtype=value_states.dtype), value_states)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[: tensor.shape[0] * attn_heads, :maxpos, :maxpos]


def baichuan_13b_gen_alibi_mask(tensor, n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head)).to(tensor.dtype)
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1).to(tensor.dtype)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    if tensor.device.type == "xpu":
        alibi_mask = alibi_mask.to(tensor.device)
    return alibi_mask


MASK_BLOCK_SIZE = 512


def baichuan_13b_get_alibi_mask(self, tensor, seq_length_with_past):
    if self.training:
        slopes = torch.Tensor(_get_interleave(self.n_head))
        position_point = (
            torch.arange(seq_length_with_past) - seq_length_with_past + 1
        )
        position_point = (
            position_point.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.n_head, seq_length_with_past, -1)
        )
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(
            -1, -2
        )
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
        mask = _buffered_future_mask(
            tensor, seq_length_with_past, alibi, self.n_head
        )
    else:
        if self.first_run:
            # Override the default max_cache_pos=4096 for memory considerations
            self.max_cache_pos = seq_length_with_past + MASK_BLOCK_SIZE
            self.first_run = False
            self.register_buffer(
                "future_mask",
                baichuan_13b_gen_alibi_mask(tensor, self.n_head, self.max_cache_pos),
                persistent=False,
            )
        if seq_length_with_past > self.max_cache_pos:
            # When max_cache_pos is not enough for current sequence length,
            # increase by MASK_BLOCK_SIZE and recalculate future_mask.
            self.max_cache_pos = seq_length_with_past + MASK_BLOCK_SIZE
            self.register_buffer(
                "future_mask",
                baichuan_13b_gen_alibi_mask(tensor, self.n_head, self.max_cache_pos),
                persistent=False,
            )
        mask = self.future_mask[
            : self.n_head, :seq_length_with_past, :seq_length_with_past
        ]
    return mask

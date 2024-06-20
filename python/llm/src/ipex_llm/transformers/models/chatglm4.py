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
# This file is adapted from
# https://huggingface.co/THUDM/chatglm2-6b-32k/blob/main/configuration_chatglm.py
#

import torch
from typing import Optional, Tuple, Union
from ipex_llm.transformers.models.utils import restore_fp8_kv_cache, update_past_key_value
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, use_sdp, use_sdp_causal
from ipex_llm.transformers.models.utils import should_use_fuse_rope, apply_rotary_pos_emb
from ipex_llm.transformers.models.chatglm2 import repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast
import math


def chatglm4_model_forward(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or\
                (past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids,
                                                 past_key_values,
                                                 padding_mask=attention_mask)

    # ipex-llm changes begin
    # 1. replace `rotary_pos_emb` with `inv_freq` and `position_ids`
    # 2. generate `causal_mask` and replace `full_attention_mask` with it
    if position_ids is None:
        if past_key_values is None:
            position_ids = torch.arange(seq_length, dtype=torch.int64, device=inputs_embeds.device)
        else:
            kv_length = past_key_values[0][0].size(2)
            position_ids = torch.arange(kv_length, kv_length + seq_length,
                                        dtype=torch.int64, device=inputs_embeds.device)
        position_ids = position_ids.repeat(batch_size, 1)

    if not getattr(self.rotary_pos_emb, "cached", False):
        rot_dim = self.rotary_pos_emb.dim
        base = 10000 * getattr(self.rotary_pos_emb, "rope_ratio", 1)
        # We should generate float inv_freq to avoid overflow, as base is too large.
        inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2,
                                                dtype=torch.float,
                                                device=inputs_embeds.device) / rot_dim))
        inv_freq = inv_freq.to(inputs_embeds.dtype)
        self.rotary_pos_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rotary_pos_emb.cached = True

    # `full_attention_mask` is not None only when
    #  `past_key_values` is not None and `seq_length` > 1
    if full_attention_mask is not None:
        causal_mask = torch.zeros([batch_size, 1, seq_length, full_attention_mask.size(-1)],
                                  dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask.masked_fill_(full_attention_mask, mask_value)
    elif self.training or (inputs_embeds.device.type != "xpu" and past_key_values is None):
        full_attention_mask = self.get_masks(input_ids,
                                             past_key_values,
                                             padding_mask=attention_mask)
        causal_mask = torch.zeros([batch_size, 1, seq_length, full_attention_mask.size(-1)],
                                  dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask.masked_fill_(full_attention_mask, mask_value)
    else:
        causal_mask = None

    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, causal_mask,
        rotary_pos_emb=(self.rotary_pos_emb.inv_freq, position_ids),
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )
    # ipex-llm changes end

    if presents is not None and type(presents) is torch.Tensor:
        presents = presents.split(1, dim=0)
        presents = list(presents)
        presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
        presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
        presents = tuple(presents)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                     if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def chatglm4_attention_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    # hidden_states: [b, sq, h]
    bsz, q_len, _ = hidden_states.size()

    # past_key_value: [bsz, n_kv_head, seq_len, head_dim]
    past_key_value = None if kv_cache is None else (kv_cache[0],
                                                    kv_cache[1])

    n_head = self.num_attention_heads_per_partition
    n_kv_head = self.num_multi_query_groups_per_partition if self.multi_query_attention else n_head
    head_dim = self.hidden_size_per_attention_head

    qkv = self.query_key_value(hidden_states)
    # [bs, q_len, np * 3 * hn] -> [bsz, n_head, seq_len, head_dim]
    qkv = qkv.view(bsz, q_len, n_head + 2 * n_kv_head, head_dim)
    qkv = qkv.transpose(1, 2)

    query_states, key_states, value_states = qkv.split([n_head,
                                                        n_kv_head,
                                                        n_kv_head], dim=1)

    kv_seq_len = key_states.shape[2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    # IPEX-LLM OPT: fuse rope
    inv_freq, position_ids = rotary_pos_emb
    rot_dim = inv_freq.size(-1) * 2
    if should_use_fuse_rope(hidden_states, rotary_pos_emb[1], self.training):
        import xe_addons
        xe_addons.rotary_two_inplaced(inv_freq, position_ids,
                                      query_states[..., :rot_dim], key_states[..., :rot_dim])
    else:
        idx_theta = torch.outer(position_ids[0].float(),
                                inv_freq.float()).to(hidden_states.dtype)
        idx_theta = idx_theta.unsqueeze(0).unsqueeze(0)
        cos = torch.cos(idx_theta).repeat_interleave(2, -1)
        sin = torch.sin(idx_theta).repeat_interleave(2, -1)
        q_rot, k_rot = apply_rotary_pos_emb(query_states[..., :rot_dim], key_states[..., :rot_dim],
                                            cos, sin, position_ids, "chatglm")
        query_states[..., :rot_dim] = q_rot[...]
        key_states[..., :rot_dim] = k_rot[...]

    # IPEX-LLM OPT: kv cache and quantize kv
    use_quantize_kv = use_quantize_kv_cache(self.query_key_value, query_states)
    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, use_quantize_kv, hidden_states.device
    )

    if use_cache:
        if past_key_value is None:
            past_key_value = torch.cat((key_states.unsqueeze(0).unsqueeze(0),
                                        value_states.unsqueeze(0).unsqueeze(0)), dim=1)
        else:
            past_key_value = (key_states, value_states)
    else:
        past_key_value = None

    # IPEX-LLM OPT: sdp
    attn_weights = None
    if use_sdp(q_len, kv_seq_len, head_dim, query_states):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states, attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states, attention_mask)
    elif use_sdp_causal(q_len, kv_seq_len, head_dim, query_states, self.training):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states, value_states,
                                                   attention_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states, value_states,
                                               attention_mask)
    elif query_states.device.type == "cpu":
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, n_head // n_kv_head)
        value_states = repeat_kv(value_states, n_head // n_kv_head)
        if q_len == kv_seq_len:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attention_mask
            )
    else:
        if use_quantize_kv:
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, n_head // n_kv_head)
        value_states = repeat_kv(value_states, n_head // n_kv_head)
        attn_weights = torch.matmul(query_states / math.sqrt(head_dim),
                                    key_states.transpose(2, 3)).to(value_states.dtype)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                   dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    # context_layer's shape: [bsz, n_head, seq_len, head_dim] -> [seq_len, bsz, n_head * head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, n_head * head_dim)
    output = self.dense(attn_output)

    return output, past_key_value

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
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import torch.nn.functional as F
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, apply_ipex_rotate_every_two
from ipex_llm.transformers.models.utils import use_sdp
from ipex_llm.transformers.models.chatglm2 import should_split_qkv_tensor
from ipex_llm.transformers.models.chatglm2 import split_tensor_along_last_dim
from transformers.modeling_outputs import BaseModelOutputWithPast


import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))
KV_CACHE_ALLOC_MIN_LENGTH = 512


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
    from ipex_llm.transformers.kv import DynamicFp8Cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    # if use_cache and use_quantize_kv_cache(
    #         self.encoder.layers[0].self_attention.query_key_value, input_ids):
    #     if not isinstance(past_key_values, DynamicFp8Cache):
    #         past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return chatglm4_model_forward_internal(
        self=self,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        full_attention_mask=full_attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


def chatglm4_model_forward_internal(
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
):
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

    use_fuse_rope = input_ids.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not self.training

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    if use_fuse_rope:
        # Repeat cos sin here, call only once for each token.
        # Chatglm2's rotary embedding is similar to gptj's, is rotate_every_two.
        # If put this to attension forward, it will generate too many times.
        cos, sin = rotary_pos_emb.split(rotary_pos_emb.shape[-1] // 2, dim=-1)
        cos = cos.squeeze(-1)
        sin = sin.squeeze(-1)
        cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        rotary_pos_emb = (cos, sin)

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )
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


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


def chatglm4_attention_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================

    # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
    device = hidden_states.device
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition,
                                       self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition,
                                     self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition,
                                                        3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    # [b, sq, np, hn] -> [b, np, sq, hn]
    query_layer, key_layer, value_layer = [k.transpose(1, 2)
                                           for k in [query_layer, key_layer, value_layer]]

    # apply relative positional encoding (rotary embedding)
    if isinstance(rotary_pos_emb, tuple) and len(rotary_pos_emb) == 2:
        # use_fuse_rope, see chatglm4_model_forward
        cos, sin = rotary_pos_emb
        rot_dim = cos.shape[-1]
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        query_layer_cur = query_layer[..., :rot_dim]
        key_layer_cur = key_layer[..., :rot_dim]
        # ipex_llm's apply_rotary_embedding can change the origin storage,
        # so query_layer will get the result directly.
        torch.ops.torch_ipex.apply_rotary_embedding(query_layer_cur, sin, cos, query_layer_cur)
        torch.ops.torch_ipex.apply_rotary_embedding(key_layer_cur, sin, cos, key_layer_cur)
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
    elif rotary_pos_emb is not None:
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    cur_length, batch_size = query_layer.shape[2], query_layer.shape[0]

    # adjust key and value for inference
    if kv_cache is not None and use_cache:
        cache_k, cache_v = kv_cache
        past_length = cache_k.size(2)

        if cache_k.stride()[1] < (past_length + cur_length) * cache_k.size(3):
            max_cache_length = past_length + cur_length + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_cache_k, new_cache_v = extend_kv_cache(batch_size,
                                                       key_layer.size(1),
                                                       self.hidden_size_per_attention_head,
                                                       past_length,
                                                       max_cache_length,
                                                       dtype=query_layer.dtype,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v

        key_layer, value_layer = append_kv_cache(cache_k, cache_v, key_layer, value_layer)

    if use_cache:
        if kv_cache is None:
            kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0),
                                  value_layer.unsqueeze(0).unsqueeze(0)), dim=1)
        else:
            kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1, -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1, -1
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )
        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1, -1,
            self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition,
            -1, -1
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] +
            (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )

    # ==================================
    # core attention computation
    # ==================================

    context_layer = core_attn_forward(query_layer, key_layer, value_layer, attention_mask)

    # =================
    # Output. [sq, b, h]
    # =================

    output = self.dense(context_layer)

    return output, kv_cache


def core_attn_forward(query_layer, key_layer, value_layer, attention_mask):
    L, S = query_layer.shape[2], key_layer.shape[2]
    if attention_mask is None and L == S:
        batch_size, n_head, seq_len, head_dim = query_layer.shape
        if should_split_qkv_tensor(query_layer, batch_size, n_head, seq_len):
            # split second dim to block size = 8
            block_size = 8
            query_split = torch.split(query_layer.to(key_layer.dtype), block_size, dim=1)
            key_split = torch.split(key_layer, block_size, dim=1)
            value_split = torch.split(value_layer, block_size, dim=1)
            results = []
            for q, k, v in zip(query_split, key_split, value_split):
                result = F.scaled_dot_product_attention(q, k, v, is_causal=True).to(k.dtype)
                results.append(result)
            context_layer = torch.cat(results, dim=1)
        else:
            context_layer = F.scaled_dot_product_attention(query_layer.to(key_layer.dtype),
                                                           key_layer,
                                                           value_layer,
                                                           is_causal=True).to(key_layer.dtype)
    else:
        # attention_mask is not None only when past_key_value is not None and q_len > 1
        if attention_mask is not None:
            attn_bias = torch.zeros(attention_mask.shape, dtype=query_layer.dtype,
                                    device=query_layer.device)
            attention_mask = ~attention_mask
            if attention_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attention_mask
        else:
            attn_bias = None

        if use_sdp(query_layer.shape[2], key_layer.shape[2],
                   query_layer.shape[-1], query_layer):
            import xe_addons
            attn_output = xe_addons.sdp(query_layer, key_layer, value_layer, attn_bias)
            context_layer = attn_output.view(query_layer.shape)
        else:
            head_dim = query_layer.size(-1)
            attn = torch.matmul(query_layer.to(key_layer.dtype),
                                key_layer.transpose(2, 3)) / math.sqrt(head_dim)
            if attn_bias is not None:
                attn += attn_bias
            attn = F.softmax(attn, dim=-1,
                             dtype=torch.float32).to(value_layer.dtype)
            context_layer = torch.matmul(attn, value_layer)
    context_layer = context_layer.transpose(1, 2).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (-1,)
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer

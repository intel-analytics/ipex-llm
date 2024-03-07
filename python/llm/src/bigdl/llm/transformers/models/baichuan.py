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
# https://huggingface.co/baichuan-inc/Baichuan-7B/blob/c1a5c7d5b7f50ecc51bb0e08150a9f12e5656756/modeling_baichuan.py
# and
# https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/a4a558127068f2ce965aa56aeb826bf501a68970/modeling_baichuan.py


import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, \
    append_kv_cache, is_enough_kv_cache_room_4_31
from bigdl.llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache
from bigdl.llm.transformers.models.utils import rotate_half, apply_rotary_pos_emb
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def baichuan_attention_forward_7b(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache(self.W_pack, hidden_states):
        forward_function = baichuan_attention_forward_7b_quantized
    else:
        forward_function = baichuan_attention_forward_7b_origin
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache
    )


def baichuan_attention_forward_7b_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    # batch_size x source_len x hidden_size
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x target_len x head_size
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x source_len x hidden_size
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    if query_states.device.type == "xpu" and not (self.training and query_states.requires_grad):
        query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                     key_states,
                                                                     position_ids,
                                                                     "baichuan")
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids, "baichuan")
    # [bsz, nh, t, hd]

    if past_key_value is None:
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            invalidInputError(
                False,
                f"Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}"
                f", but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            invalidInputError(
                attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                f"but is {attention_mask.size()}"
            )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        kv_seq_len = key_states.shape[-2]
        if use_cache:
            k_cache, v_cache = init_fp8_kv_cache(
                bsz, self.num_heads, kv_seq_len, self.head_dim,
                device=device
            )
            key_states, value_states = append_kv_cache(k_cache, v_cache, key_states, value_states)
            past_key_value = (key_states, value_states)
    else:
        k_cache, v_cache = past_key_value
        key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                       key_states, value_states)
        kv_seq_len = key_states.shape[-2]
        past_key_value = (key_states, value_states)
        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        else:
            import linear_q4_0
            attn_weights = linear_q4_0.query_key_fp8_matmul(query_states, key_states)

        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            invalidInputError(
                False,
                f"Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}"
                f", but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            invalidInputError(
                attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                f"but is {attention_mask.size()}"
            )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import linear_q4_0
            attn_output = linear_q4_0.attn_value_fp8_matmul(attn_weights,
                                                            value_states.transpose(-1, -2))

    invalidInputError(
        attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim),
        f"`attn_output` should be of size "
        f"{(bsz, self.num_heads, q_len, self.head_dim)},"
        f"but is {attn_output.size()}"
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def baichuan_attention_forward_7b_origin(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    # batch_size x source_len x hidden_size
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x target_len x head_size
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x source_len x hidden_size
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    enough_kv_room = True
    if past_key_value is not None:
        enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value, seq_len=kv_seq_len)
        kv_seq_len += past_key_value[0].shape[-2]
    if query_states.device.type == "xpu" and not (self.training and query_states.requires_grad):
        query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                     key_states,
                                                                     position_ids,
                                                                     "baichuan")
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids, "baichuan")
    # [bsz, nh, t, hd]

    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if past_key_value is not None:
        # reuse k, v, self_attention
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if not enough_kv_room:
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v

        key_states, value_states = append_kv_cache(cache_k, cache_v, key_states, value_states)

    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        new_key_states, new_value_states = init_kv_cache(bsz,
                                                         self.num_heads,
                                                         self.head_dim,
                                                         kv_seq_len,
                                                         max_cache_length,
                                                         dtype=key_states.dtype,
                                                         device=device)
        new_key_states[:] = key_states
        new_value_states[:] = value_states
        key_states = new_key_states
        value_states = new_value_states

    past_key_value = (key_states, value_states) if use_cache else None

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        invalidInputError(False,
                          f"Attention weights should be of size "
                          f"{(bsz, self.num_heads, q_len, kv_seq_len)}"
                          f", but is {attn_weights.size()}")

    if attention_mask is not None:
        invalidInputError(attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                          f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                          f"but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    invalidInputError(attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim),
                      f"`attn_output` should be of size "
                      f"{(bsz, self.num_heads, q_len, self.head_dim)},"
                      f"but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2)
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
    if use_quantize_kv_cache(self.W_pack, hidden_states):
        forward_function = baichuan_attention_forward_13b_quantized
    else:
        forward_function = baichuan_attention_forward_13b_origin
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache
    )


def baichuan_attention_forward_13b_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if past_key_value is None:
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        kv_seq_len = key_states.shape[-2]
        if use_cache:
            k_cache, v_cache = init_fp8_kv_cache(
                bsz, self.num_heads, kv_seq_len, self.head_dim,
                device=device
            )
            key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                           key_states, value_states)
            past_key_value = (key_states, value_states)
    else:
        k_cache, v_cache = past_key_value
        key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                       key_states, value_states)
        kv_seq_len = key_states.shape[-2]
        past_key_value = (key_states, value_states)
        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        else:
            import linear_q4_0
            attn_weights = linear_q4_0.query_key_fp8_matmul(query_states, key_states)

        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import linear_q4_0
            attn_output = linear_q4_0.attn_value_fp8_matmul(attn_weights,
                                                            value_states.transpose(-1, -2))

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def baichuan_attention_forward_13b_origin(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    enough_kv_room = True
    if past_key_value is not None:
        enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value, seq_len=kv_seq_len)
        kv_seq_len += past_key_value[0].shape[-2]

    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if past_key_value is not None:
        # reuse k, v, self_attention
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if not enough_kv_room:
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v

        key_states, value_states = append_kv_cache(cache_k, cache_v, key_states, value_states)

    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        new_key_states, new_value_states = init_kv_cache(bsz,
                                                         self.num_heads,
                                                         self.head_dim,
                                                         kv_seq_len,
                                                         max_cache_length,
                                                         dtype=key_states.dtype,
                                                         device=device)
        new_key_states[:] = key_states
        new_value_states[:] = value_states
        key_states = new_key_states
        value_states = new_value_states

    past_key_value = (key_states, value_states) if use_cache else None

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        if q_len == 1:  # inference with cache
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, -1:, :]
            else:
                attention_mask = attention_mask[:, -1:, :]
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

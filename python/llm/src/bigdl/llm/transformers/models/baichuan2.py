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
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import rotate_half, apply_rotary_pos_emb
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu
from transformers.utils import logging, ContextManagers
from bigdl.llm.transformers.models.llama import get_ipex_version
logger = logging.get_logger(__name__)

try:
    from xformers import ops as xops
except ImportError:
    xops = None
    logger.warning(
        "Xformers is not installed correctly. If you want to use memory_efficient_attention to "
        "accelerate training use the following command to install Xformers\npip install xformers."
    )


KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def baichuan_13b_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu" and not (self.training and hidden_states.requires_grad):
        if get_ipex_version() <= "2.0.110+xpu":
            hidden_states, _ = torch.ops.torch_ipex.rms_norm(hidden_states,
                                                             [self.weight.size(0)],
                                                             self.weight)
        else:
            hidden_states = torch.ops.torch_ipex.fast_rms_norm(hidden_states,
                                                               [self.weight.size(0)],
                                                               self.weight,
                                                               None,
                                                               self.epsilon)
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states.to(input_dtype)
    return hidden_states


def baichuan_attention_forward_7b(
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

    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if past_key_value is not None:
        # reuse k, v, self_attention
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
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

    if xops is not None and self.training:
        attn_weights = None
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = xops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask()
        )
    else:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True,
                                            enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                         attn_mask=attention_mask)
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
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = (
        proj.unflatten(-1, (3, self.hidden_size))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
    )
    query_states = (
        proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    key_states = (
        proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    value_states = (
        proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if past_key_value is not None:
        # reuse k, v, self_attention
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            if device.type == 'xpu':
                torch.xpu.empty_cache()
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
    if xops is not None and self.training:
        attn_weights = None
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)
        # attn_output = xops.memory_efficient_attention(
        #     query_states, key_states, value_states, attn_bias=attention_mask
        # )
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True,
                                            enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                         attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2)
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

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
# https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/7ab7b3c18eb8e5232ce2a3f720d4e6f4b53a2806/yuan_hf_model.py
# which is licensed under Apache License 2.0:
#
# https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/7ab7b3c18eb8e5232ce2a3f720d4e6f4b53a2806/README.md#%E5%A3%B0%E6%98%8E%E4%B8%8E%E5%8D%8F%E8%AE%AEterms-and-conditions
#

import copy
import math
from einops import rearrange
from typing import Optional, Tuple

import torch
import torch.nn as nn

from bigdl.llm.transformers.models.llama import should_use_fuse_rope
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb, \
    apply_rotary_pos_emb_no_cache_xpu, mlp_fusion_check, fp16_fusion_check
from bigdl.llm.transformers.models.llama import should_use_fuse_rope
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import is_enough_kv_cache_room_4_31
from bigdl.llm.transformers.low_bit_linear import SYM_INT4, FP8E5

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def use_decoding_fast_path(q_type, use_fuse_rope, enough_kv_room, bs):
    return q_type in [SYM_INT4, FP8E5] and \
        use_fuse_rope and enough_kv_room and bs == 1


def should_use_fuse_rope(self, query_states, position_ids):
    use_fuse_rope = query_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and query_states.requires_grad)
    use_fuse_rope = use_fuse_rope and position_ids is not None
    return use_fuse_rope


def yuan_mlp_forward(
    self,
    x: torch.Tensor,
    residual=None
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    bsz, hidden_size = x_2d.shape
    qtype = getattr(self.up_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training):
        import linear_q4_0
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out = self.down_proj(linear_q4_0.mlp_forward_xpu(
            x_2d, self.up_proj.weight.data, self.gate_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.up_proj.out_len,
            qtype
        ))
        if residual is not None:
            return out + residual
        else:
            return out
    elif fp16_fusion_check(self.up_proj, x, self.training) and \
            hidden_size == 4096 and bsz == 1:
        hidden_states1 = torch.ops.torch_ipex.mm_silu(x, self.up_proj.weight)
        hidden_states = torch.ops.torch_ipex.mm_resmul(
            x, self.gate_proj.weight, hidden_states1
        )
        if residual is None:
            hidden_states = torch.matmul(hidden_states, self.down_proj.weight)
        else:
            attn_output = torch.addmm(
                residual.flatten(0, -2),
                hidden_states.flatten(0, -2),
                self.down_proj.weight,
                beta=1,
            )
            hidden_states = attn_output.view(x.shape)
        return hidden_states
    else:
        out = self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))
        if residual is not None:
            return out + residual
        else:
            return out


def yuan_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    before_hidden_states = None
    is_first_step = False
    self.use_shareqk = False

    enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value)

    if use_cache:
        if past_key_value is None:
            inference_hidden_states_memory = torch.empty(bsz, 2,
                                                         hidden_states.shape[2],
                                                         dtype=hidden_states.dtype)
            is_first_step = True
        else:
            before_hidden_states = past_key_value[2]

    if use_cache:
        if is_first_step:
            if q_len >= 2:
                inference_hidden_states_memory = hidden_states[:, -2:, :]
            else:
                inference_hidden_states_memory[:, :, :] = 0
                inference_hidden_states_memory[:, -1:, :] = hidden_states[:, -1:, :]
        else:
            hidden_states_tmp = before_hidden_states[:, -1:, :]
            inference_hidden_states_memory = \
                copy.deepcopy(torch.cat((hidden_states_tmp, hidden_states), dim=1))

    value_states = \
        self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if self.use_shareqk:
        # use_shareqk is disabled for now
        qk_states = self.qk_proj(hidden_states).view(bsz, q_len, self.num_heads*self.head_dim)
        query_key = qk_states.unsqueeze(2) * self.qk_weight + self.qk_bias
        query_states, key_states = torch.unbind(query_key, dim=2)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    else:
        hidden_states = self.lf_gate(hidden_states, before_hidden_states)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        qk_states = torch.cat([query_states, key_states], dim=-1)
        qk_states = qk_states.view(bsz, q_len,
                                   self.num_heads,
                                   int(qk_states.shape[-1]//self.num_heads))
        (query_states, key_states) = torch.chunk(qk_states, 2, dim=-1)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if use_fuse_rope:
        query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                     key_states,
                                                                     position_ids,
                                                                     "yuan")
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states,
                                                        cos, sin,
                                                        position_ids,
                                                        "yuan")

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
                                                       dtype=cache_k.d_type,
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

    past_key_value = \
        (key_states, value_states, inference_hidden_states_memory) if use_cache else None

    attn_weights = \
        torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    invalidInputError(attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len),
                      "Attention weights should be of size "
                      f"{(bsz, self.num_heads, q_len, kv_seq_len)}, "
                      f"but is {attn_weights.size()}")

    if attention_mask is not None:
        invalidInputError(attention_mask.size() == (bsz, 1, q_len, kv_seq_len),
                          f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                          f"but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights,
                                 torch.tensor(torch.finfo(attn_weights.dtype).min))

    # upcast attention to fp32
    attn_weights = \
        torch.nn.functional.softmax(attn_weights,
                                    dim=-1,
                                    dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    invalidInputError(attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim),
                      "`attn_output` should be of size "
                      f"{(bsz, self.num_heads, q_len, self.head_dim)}, "
                      f"but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

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

from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb, \
    apply_rotary_pos_emb_cache_freq_xpu, mlp_fusion_check, fp16_fusion_check
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache
from ipex_llm.transformers.models.utils import is_enough_kv_cache_room_4_31, SILU

import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def should_use_fuse_rope(self, hidden_states, position_ids):
    use_fuse_rope = hidden_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and hidden_states.requires_grad)
    use_fuse_rope = use_fuse_rope and position_ids is not None
    return use_fuse_rope


def yuan_localized_filtering_forward(
    self,
    inputs: torch.Tensor,
    before_hidden_states: torch.Tensor,
    dtype: torch.dtype,
):
    if self.conv1.weight.dtype != torch.half:
        self.half()

    invalidInputError(self.lf_conv2d_num_pad == 1, "padding must be 1")
    invalidInputError(not self.training, ("training is not supported for now, "
                                          "please call model.eval() before inference"))
    if before_hidden_states is None:
        inputs = inputs.half()
        lf_output = self._inference_forward(inputs, None)
    else:
        # only change next token logic
        bsz, seq_len, embed_dim = inputs.size()
        seq_len_before, _, _ = before_hidden_states.size()
        invalidInputError(seq_len == 1 and seq_len_before == 3,
                          f"wrong sequence length: {seq_len} {seq_len_before}")

        residual = before_hidden_states[-1:, :, :]
        inputs = before_hidden_states.view(3, 1, bsz, embed_dim).permute(2, 3, 0, 1)

        output1 = self.conv1(inputs)
        output2 = self.conv2(output1[:, :, 1:-1, :])
        output2 = output2[:, :, 1:-1, :]
        output2 = output2.view(1, bsz, embed_dim)

        invalidInputError(output2.shape == residual.shape,
                          f"wrong shape: {output2.shape} {residual.shape}")

        lf_output = self.output_layernorm(output2 + residual)
        lf_output = lf_output.transpose(0, 1)

    lf_output = lf_output.to(dtype)
    return lf_output


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
            SILU, qtype
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
    if use_quantize_kv_cache(self.merged_q_proj, hidden_states):
        forward_function = yuan_attention_forward_quantized
    else:
        forward_function = yuan_attention_forward_origin
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )


def yuan_attention_forward_quantized(
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
    before_hidden_states = None
    is_first_step = False

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)

    invalidInputError(use_cache, "use_cache=True is needed")
    invalidInputError(not self.use_shareqk, "use_shareqk is not supported for now")

    if past_key_value is None:
        is_first_step = True
        if q_len >= 2:
            before_hidden_states = hidden_states[:, -2:, :].transpose(0, 1).half()
        else:
            before_hidden_states = torch.zeros(2, bsz, self.hidden_size,
                                               dtype=torch.half, device=hidden_states.device)
            before_hidden_states[-1:, :, :] = hidden_states[:, -1:, :].transpose(0, 1)
    else:
        before_hidden_states = past_key_value[2]
        this_hidden_states = torch.cat([
            before_hidden_states,
            hidden_states.transpose(0, 1).half(),
        ], dim=0)
        before_hidden_states = this_hidden_states[-2:, :, ]

    value_states = \
        self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if is_first_step:
        hidden_states = yuan_localized_filtering_forward(self.lf_gate, hidden_states,
                                                         None, hidden_states.dtype)
    else:
        hidden_states = yuan_localized_filtering_forward(self.lf_gate, hidden_states,
                                                         this_hidden_states, hidden_states.dtype)
    query_states = self.merged_q_proj(hidden_states)
    key_states = self.merged_k_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    if use_fuse_rope:
        query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(query_states,
                                                                       key_states,
                                                                       sin, cos,
                                                                       "yuan",
                                                                       position_ids)
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states,
                                                        cos, sin,
                                                        position_ids,
                                                        "yuan")

    if past_key_value is None:
        # should use origin attn here
        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if use_cache:
            k_cache, v_cache = init_fp8_kv_cache(
                bsz, self.num_heads, kv_seq_len, self.head_dim, device=device
            )
            key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                           key_states, value_states)
            past_key_value = (key_states, value_states, before_hidden_states)

    else:
        k_cache, v_cache, _ = past_key_value
        key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                       key_states, value_states)
        past_key_value = (key_states, value_states, before_hidden_states)

        # torch.matmul
        if query_states.size(2) != 1 or device.type != 'xpu':
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        else:
            import linear_q4_0
            attn_weights = linear_q4_0.query_key_fp8_matmul(query_states, key_states)

        attn_weights = attn_weights / math.sqrt(self.head_dim)

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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        if query_states.size(2) != 1 or device.type != 'xpu':
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import linear_q4_0
            attn_output = linear_q4_0.attn_value_fp8_matmul(attn_weights, value_states)

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


def yuan_attention_forward_origin(
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

    invalidInputError(use_cache, "use_cache=True is needed")
    invalidInputError(not self.use_shareqk, "use_shareqk is not supported for now")

    if past_key_value is None:
        is_first_step = True
        if q_len >= 2:
            before_hidden_states = hidden_states[:, -2:, :].transpose(0, 1).half()
        else:
            before_hidden_states = torch.zeros(2, bsz, self.hidden_size,
                                               dtype=torch.half, device=hidden_states.device)
            before_hidden_states[-1:, :, :] = hidden_states[:, -1:, :].transpose(0, 1)
    else:
        before_hidden_states = past_key_value[2]
        this_hidden_states = torch.cat([
            before_hidden_states,
            hidden_states.transpose(0, 1).half(),
        ], dim=0)
        before_hidden_states = this_hidden_states[-2:, :, ]

    value_states = \
        self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if is_first_step:
        hidden_states = yuan_localized_filtering_forward(self.lf_gate, hidden_states,
                                                         None, hidden_states.dtype)
    else:
        hidden_states = yuan_localized_filtering_forward(self.lf_gate, hidden_states,
                                                         this_hidden_states, hidden_states.dtype)
    query_states = self.merged_q_proj(hidden_states)
    key_states = self.merged_k_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    if use_fuse_rope:
        query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(query_states,
                                                                       key_states,
                                                                       sin, cos,
                                                                       "yuan",
                                                                       position_ids)
    else:
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

    past_key_value = \
        (key_states, value_states, before_hidden_states) if use_cache else None

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

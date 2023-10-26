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
# ===========================================================================
#
# This file is copied from
# https://github.com/mit-han-lab/streaming-llm/blob/main/streaming_llm/pos_shift/modify_falcon.py
# which is licensed under the MIT license:
#
# MIT License
#
# Copyright (c) 2023 MIT HAN Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import types
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    rotate_half,
)

__all__ = ["enable_falcon_pos_shift_attention"]


def falcon_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    fused_qkv = self.query_key_value(
        hidden_states
    )  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(
        batch_size * self.num_heads, q_length, self.head_dim
    )

    # dirty hack to fix the inconsistency between falcon-40b and falcon-7b
    num_kv = self.num_heads if self.num_heads == 128 else self.num_kv
    key_layer = key_layer.transpose(1, 2).reshape(
        batch_size * num_kv,
        q_length,
        self.head_dim,
    )
    value_layer = value_layer.transpose(1, 2).reshape(
        batch_size * num_kv, q_length, self.head_dim
    )

    past_len = 0
    if layer_past is not None:
        past_len = layer_past[0].shape[1]

    query_layer_copy = query_layer.clone()
    query_layer, _ = self.maybe_rotary(query_layer, query_layer_copy, past_len)
    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, head_dim, kv_length]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None

    key_layer_copy = key_layer.clone()
    _, key_layer = self.maybe_rotary(key_layer_copy, key_layer, 0)

    _, kv_length, _ = key_layer.shape

    if alibi is None:
        query_layer_ = query_layer.reshape(
            batch_size, self.num_heads, -1, self.head_dim
        )
        key_layer_ = key_layer.reshape(batch_size, num_kv, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv, -1, self.head_dim)

        if layer_past is not None:
            attn_output = F.scaled_dot_product_attention(
                query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True
            )

        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        outputs = (output_tensor, present)
        assert not output_attentions  # not supported.
        return outputs
    else:
        attention_mask_float = (
            (attention_mask * 1.0).masked_fill(attention_mask, -1e9).to(torch.bfloat16)
        )
        matmul_result = query_layer @ key_layer.transpose(-1, -2)

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(
            batch_size, self.num_heads, q_length, kv_length
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        # attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(
            (attention_scores + alibi.view(batch_size, self.num_heads, 1, -1))
            * self.inv_norm_factor
            + attention_mask_float,
            dim=-1,
            dtype=hidden_states.dtype,
        )
        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads, q_length, kv_length
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = attention_probs_reshaped @ value_layer

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


def enable_falcon_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_falcon_pos_shift_attention(
                module,
            )

        if "self_attention" == name[-14:]:
            model._modules[name].forward = types.MethodType(
                falcon_pos_shift_attention_forward, model._modules[name]
            )

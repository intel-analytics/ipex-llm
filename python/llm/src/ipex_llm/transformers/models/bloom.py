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
# https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/bloom/modeling_bloom.py
# which is licensed under Apache License 2.0:
#
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch.nn import functional as F
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache

import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool):
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bloom_attention_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
        head_mask: Optional[torch.Tensor]=None,
        use_cache: bool=False,
        output_attentions: bool=False,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(
        batch_size * self.num_heads,
        q_length,
        self.head_dim
    )
    key_layer = key_layer.permute(0, 2, 3, 1).reshape(
        batch_size * self.num_heads,
        self.head_dim,
        q_length
    )
    value_layer = value_layer.transpose(1, 2).reshape(
        batch_size * self.num_heads,
        q_length,
        self.head_dim
    )
    _, _, kv_length = key_layer.shape
    if layer_past is not None:
        kv_length += layer_past[0].shape[-1]
    query_layer = query_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).view(batch_size, self.num_heads, q_length, self.head_dim)
    value_layer = value_layer.view(batch_size, self.num_heads, q_length, self.head_dim)
    device = hidden_states.device
    if layer_past is not None:
        # reuse k, v, self_attention
        cache_k = layer_past[0].transpose(1, 2).view(batch_size, self.num_heads, -1, self.head_dim)
        cache_v = layer_past[1].view(batch_size, self.num_heads, -1, self.head_dim)
        if cache_k.stride()[1] < kv_length * cache_k.size(3):
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(
                batch_size,
                self.num_heads,
                self.head_dim,
                cache_k.size(2),
                kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH,
                dtype=cache_k.dtype,
                device=device
            )
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v

        key_layer, value_layer = append_kv_cache(cache_k, cache_v, key_layer, value_layer)

    elif use_cache:
        max_cache_length = kv_length + KV_CACHE_ALLOC_BLOCK_LENGTH
        new_key_states, new_value_states = init_kv_cache(
            batch_size,
            self.num_heads,
            self.head_dim,
            kv_length,
            max_cache_length,
            dtype=key_layer.dtype,
            device=device
        )
        new_key_states[:] = key_layer
        new_value_states[:] = value_layer
        key_layer = new_key_states
        value_layer = new_value_states

    query_layer = query_layer.view(batch_size*self.num_heads, -1, self.head_dim)
    key_layer = key_layer.view(batch_size*self.num_heads, -1, self.head_dim).transpose(1, 2)
    value_layer = value_layer.view(batch_size*self.num_heads, -1, self.head_dim)
    _, _, kv_length = key_layer.shape
    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None

    # [batch_size * num_heads, q_length, kv_length]
    # we use `torch.Tensor.baddbmm`
    # instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
    matmul_result = alibi.baddbmm(
        batch1=query_layer,
        batch2=key_layer,
        beta=self.beta,
        alpha=self.inv_norm_factor,
    )

    # change view to [batch_size, num_heads, q_length, kv_length]
    attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

    # cast attention scores to fp32,
    # compute scaled softmax and cast back to initial dtype
    # - [batch_size, num_heads, q_length, kv_length]
    input_dtype = attention_scores.dtype
    # `float16` has a minimum value of -65504.0,
    # whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
    if input_dtype == torch.float16:
        attention_scores = attention_scores.to(torch.float)
    attn_weights = torch.masked_fill(
        attention_scores,
        attention_mask,
        torch.finfo(attention_scores.dtype).min
    )
    attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

    # [batch_size, num_heads, q_length, kv_length]
    attention_probs = self.attention_dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    # change view [batch_size x num_heads, q_length, kv_length]
    attention_probs_reshaped = attention_probs.view(
        batch_size * self.num_heads,
        q_length,
        kv_length
    )

    # matmul: [batch_size * num_heads, q_length, head_dim]
    context_layer = torch.bmm(attention_probs_reshaped, value_layer)

    # change view [batch_size, q_length, num_heads * head_dim]
    context_layer = self._merge_heads(context_layer)

    # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + F.linear(
                context_layer[:, :, int(i * slices): int((i + 1) * slices)],
                self.dense.weight[:, int(i * slices): int((i + 1) * slices)],
            )
    else:
        output_tensor = self.dense(context_layer)

    output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

    outputs = (output_tensor, present)
    if output_attentions:
        outputs += (attention_probs,)

    return outputs

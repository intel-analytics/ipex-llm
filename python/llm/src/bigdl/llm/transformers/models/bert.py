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
# https://github.com/huggingface/transformers/blob/v4.38.0/src/transformers/models/bert/modeling_bert.py
# which is licensed under Apache License 2.0:
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import math
import torch
from typing import Optional, Tuple
from transformers.models.bert.modeling_bert import BertSelfAttention, BertEncoder
from bigdl.llm.utils.common import invalidInputError


def merge_qkv(module: torch.nn.Module):
    if isinstance(module, BertSelfAttention):
        q_w = module.query.weight.data
        k_w = module.key.weight.data
        v_w = module.value.weight.data
        q_b = module.query.bias.data
        k_b = module.key.bias.data
        v_b = module.value.bias.data
        new_w = torch.cat([q_w, k_w, v_w], dim=0)
        new_b = torch.cat([q_b, k_b, v_b], dim=-1)
        qkv = torch.nn.Linear(0, 0, bias=True)
        qkv.weight = torch.nn.Parameter(new_w, requires_grad=False)
        qkv.bias = torch.nn.Parameter(new_b, requires_grad=False)
        qkv.in_features = module.query.in_features
        qkv.out_features = module.query.out_features * 3
        module.qkv = qkv
        del module.query
        del module.key
        del module.value


def self_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    output_attentions: Optional[bool] = False,
):
    invalidInputError(encoder_hidden_states is None,
                      "cross attention is not supported")
    invalidInputError(not self.is_decoder,
                      "bert decoder is not supported")
    invalidInputError(self.position_embedding_type == "absolute",
                      "relative query/key is not supported")

    qkv_output = self.qkv(hidden_states)
    (query_layer, key_layer, value_layer) = torch.chunk(qkv_output, 3, -1)
    query_layer = self.transpose_for_scores(query_layer)
    key_layer = self.transpose_for_scores(key_layer)
    value_layer = self.transpose_for_scores(value_layer)

    if past_key_value is not None:
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size),
                                    key_layer.transpose(-1, -2))

    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs


def encoder_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    return_dict: Optional[bool] = True,
):
    if not attention_mask.any():
        attention_mask = None
    return BertEncoder.forward(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

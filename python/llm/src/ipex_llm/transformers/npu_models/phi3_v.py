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
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
# which is licensed under Apache License 2.0:
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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


import torch
import importlib
from torch import nn
from typing import Optional, Tuple, List
from transformers.models.clip.modeling_clip import CLIPAttention
from ipex_llm.utils.common.log4Error import invalidInputError


def merge_qkv(module: torch.nn.Module):
    if isinstance(module, CLIPAttention):
        new_weight = torch.cat([
            module.q_proj.weight.data,
            module.k_proj.weight.data,
            module.v_proj.weight.data,
        ], dim=0)

        if module.q_proj.bias is not None:
            qkv_proj = torch.nn.Linear(0, 0, bias=True)
            new_bias = torch.cat([
                module.q_proj.bias.data,
                module.k_proj.bias.data,
                module.v_proj.bias.data,
            ], dim=0)
            qkv_proj.bias = torch.nn.Parameter(new_bias, requires_grad=False)
        else:
            qkv_proj = torch.nn.Linear(0, 0, bias=False)
        qkv_proj.weight = torch.nn.Parameter(new_weight, requires_grad=False)
        qkv_proj.in_features = new_weight.size(1)
        qkv_proj.out_features = new_weight.size(0)
        module.qkv_proj = qkv_proj

        del module.q_proj, module.k_proj, module.v_proj


def phi3v_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    # ipex-llm changes start
    from ipex_llm.transformers.kv import DynamicNormalCache
    # IPEX-LLM OPT: kv cache and quantize kv cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if use_cache:
        if not isinstance(past_key_values, DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    modeling_module_name = self.__class__.__module__
    module = importlib.import_module(modeling_module_name)
    return module.Phi3VModel.forward(
        self=self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


def phi3v_encoder_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, tgt_len, embed_dim = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, tgt_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = query_states.reshape(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        invalidInputError(
            False,
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)},"
            f" but is {attn_weights.size()}"
        )

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            invalidInputError(
                False,
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                f" {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) \
            + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            invalidInputError(
                False,
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)},"
                f" but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        # this operation is a bit akward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        invalidInputError(
            False,
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)},"
            f" but is {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped

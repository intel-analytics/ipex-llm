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
# https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/phi/modeling_phi.py
# which is licensed under Apache License 2.0:
#
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

from ipex_llm.transformers.models.utils import apply_rotary_pos_emb_cache_freq_xpu
from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.utils.common.log4Error import invalidInputError

from typing import Optional, Tuple, List
from transformers.cache_utils import Cache
from transformers.models.phi.modeling_phi import repeat_kv, apply_rotary_pos_emb
from transformers.models.phi.modeling_phi import PhiModel


def should_use_fuse_rope(self, hidden_states, position_ids):
    use_fuse_rope = (
        hidden_states.device.type == "xpu" and
        hidden_states.numel() == hidden_states.size(-1) and
        not (self.training and hidden_states.requires_grad) and
        position_ids is not None
    )
    return use_fuse_rope


def merge_qkv(module: torch.nn.Module):
    if module.__class__.__name__ == "PhiAttention":
        new_weight = torch.cat([
            module.q_proj.weight.data,
            module.k_proj.weight.data,
            module.v_proj.weight.data,
        ], dim=0)
        new_bias = torch.cat([
            module.q_proj.bias.data,
            module.k_proj.bias.data,
            module.v_proj.bias.data,
        ], dim=-1)

        qkv_proj = torch.nn.Linear(0, 0, bias=True)
        qkv_proj.weight = torch.nn.Parameter(new_weight, requires_grad=False)
        qkv_proj.bias = torch.nn.Parameter(new_bias, requires_grad=False)
        qkv_proj.in_features = new_weight.size(1)
        qkv_proj.out_features = new_weight.size(0)
        module.qkv_proj = qkv_proj

        del module.q_proj, module.k_proj, module.v_proj


def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    invalidInputError(not self.qk_layernorm, "`qk_layernorm` must be false")

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim:],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim:],
    )

    # IPEX-LLM OPT: fuse rope
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)

    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    if use_fuse_rope:
        query_rot, key_rot = apply_rotary_pos_emb_cache_freq_xpu(query_rot, key_rot, sin,
                                                                 cos, "stablelm", position_ids)
    else:
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    invalidInputError(past_key_value is not None,
                      "`past_key_value` cannot be None")
    key_states, value_states = past_key_value.update(key_states, value_states,
                                                     self.layer_idx, None, new_layout=True)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
    attn_weights = torch.matmul(
        query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                               dtype=torch.float32).to(value_states.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                               training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.dense(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    # IPEX-LLM OPT: kv cache but no sdp (its head_dim 80, cannot use sdp)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if use_cache:
        if not isinstance(past_key_values, DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    return PhiModel.forward(
        self=self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

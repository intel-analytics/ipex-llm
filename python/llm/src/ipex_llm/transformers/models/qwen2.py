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
# https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/qwen2/modeling_qwen2.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import os
from typing import Optional, Tuple, Union, List

import torch
from torch.nn import CrossEntropyLoss

from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import SILU, mlp_fusion_check
from ipex_llm.transformers.models.utils import should_use_fuse_rope
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, \
    should_use_compresskv, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicNormalCache, \
    DynamicCompressCache, DynamicCompressFp8Cache

from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2Attention, Qwen2MLP
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache


def qwen2_model_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # IPEX-LLM OPT start: kv cache and quantize kv cache
    inputs = input_ids if input_ids is not None else inputs_embeds
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    use_cache = True if inputs.device.type == "xpu" else use_cache

    use_quantize_kv = self.config.hidden_size != 3584 and use_quantize_kv_cache(
        self.layers[0].mlp.down_proj, inputs,
        self.config.num_attention_heads, self.config.num_key_value_heads
    )

    use_compress_kv = should_use_compresskv(inputs, inputs.shape[1]) or \
        isinstance(past_key_values, DynamicCompressCache)

    if use_cache:
        if use_compress_kv and not isinstance(past_key_values, DynamicCompressCache):
            if use_quantize_kv:
                past_key_values = DynamicCompressFp8Cache.from_legacy_cache(past_key_values)
            else:
                past_key_values = DynamicCompressCache.from_legacy_cache(past_key_values)
        elif use_quantize_kv and not use_compress_kv and not isinstance(past_key_values,
                                                                        DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
        if not use_quantize_kv and not use_compress_kv and not isinstance(past_key_values,
                                                                          DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    # ipex-llm changes end

    # `cache_position` is required after transformers 4.42
    if cache_position is not None:
        kwargs = {"cache_position": cache_position}
    else:
        kwargs = {}

    return Qwen2Model.forward(
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
        **kwargs
    )


def qwen2_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,  # for transformers >= 4.42
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    # ipex-llm changes start: remove `logits.float()` to reduce memory usage with long input
    # logits = logits.float()
    # ipex-llm changes end

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, Qwen2Attention)
    if isinstance(module, Qwen2Attention) and os.environ.get("IPEX_LLM_LOW_MEM", None) == "1":
        del module.rotary_emb.cos_cached
        del module.rotary_emb.sin_cached


def padding_mlp(module: torch.nn.Module):
    # for qwen 1.5 14B
    if isinstance(module, Qwen2MLP):
        hidden_size = module.gate_proj.weight.shape[1]
        intermediate_size = module.gate_proj.weight.shape[0]
        padding_intermediate_size = (intermediate_size + 256 - 1) // 256 * 256
        if intermediate_size % 256 == 0:
            return

        gate_weight = module.gate_proj.weight.data
        new_gate_weight = torch.zeros([padding_intermediate_size, hidden_size],
                                      dtype=gate_weight.dtype, device=gate_weight.device)
        new_gate_weight[:intermediate_size, :] = gate_weight
        if hasattr(module.gate_proj, 'out_features'):
            module.gate_proj.out_features = padding_intermediate_size
        module.gate_proj.weight = torch.nn.Parameter(new_gate_weight, requires_grad=False)

        up_weight = module.up_proj.weight.data
        new_up_weight = torch.zeros([padding_intermediate_size, hidden_size],
                                    dtype=up_weight.dtype, device=up_weight.device)
        new_up_weight[:intermediate_size, :] = up_weight
        if hasattr(module.gate_proj, 'out_features'):
            module.up_proj.out_features = padding_intermediate_size
        module.up_proj.weight = torch.nn.Parameter(new_up_weight, requires_grad=False)

        down_weight = module.down_proj.weight.data
        new_down_weight = torch.zeros([hidden_size, padding_intermediate_size],
                                      dtype=down_weight.dtype, device=down_weight.device)
        new_down_weight[:, :intermediate_size] = down_weight
        if hasattr(module.gate_proj, 'out_features'):
            module.down_proj.in_features = padding_intermediate_size
        module.down_proj.weight = torch.nn.Parameter(new_down_weight, requires_grad=False)


def qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    # [CompressKV]
    from ipex_llm.transformers.kv import DynamicCompressCache
    use_compresskv = isinstance(past_key_value, DynamicCompressCache)

    if hasattr(self, 'qkv_proj') and self.qkv_proj is not None:
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        qkv = qkv.transpose(1, 2)
        query_states, key_states, value_states = qkv.split([self.num_heads,
                                                            self.num_key_value_heads,
                                                            self.num_key_value_heads], dim=1)
    else:
        # when quant_method is 'gptq'
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim) \
                               .transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim) \
                                   .transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos, sin = cos.to(device), sin.to(device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids)

    if past_key_value is not None:
        # [CompressKV]
        if use_compresskv:
            enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx,
                                                          q_len)
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx,
                query_states, attention_mask, self.num_key_value_groups,
                self.config, enough_kv_room, 256)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states,
                                                             self.layer_idx, None)

    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == kv_seq_len
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def qwen2_mlp_forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    qtype = getattr(self.gate_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training):
        import xe_linear
        return self.down_proj(xe_linear.mlp_forward_xpu(
            x_2d, self.gate_proj.weight.data, self.up_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.gate_proj.out_len,
            SILU, qtype
        ))
    elif x.device.type == "xpu" and not self.training:
        import xe_addons
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        xe_addons.mlp_silu_mul_inplaced(gate, up)
        return self.down_proj(gate)
    else:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# which is licensed under Apache License 2.0:
#
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

import torch

from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import make_cache_contiguous_inplaced
from ipex_llm.transformers.models.utils import use_quantize_kv_cache
from ipex_llm.transformers.models.utils import should_use_compresskv, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.kv import DynamicNormalCache, DynamicFp8Cache
from ipex_llm.transformers.kv import DynamicCompressCache, DynamicCompressFp8Cache


def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
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
    use_quantize_kv = use_quantize_kv_cache(
        self.layers[0].mlp.down_proj, inputs,
        self.config.num_attention_heads, self.config.num_key_value_heads
    )
    use_compresskv = should_use_compresskv(inputs, inputs.shape[1]) or \
        isinstance(past_key_values, DynamicCompressCache)
    # disable llama3.2 1b for prefill performance and output quality
    use_compresskv = use_compresskv and self.config.hidden_size != 2048
    if use_cache:
        if use_compresskv and not isinstance(past_key_values, DynamicCompressCache):
            if use_quantize_kv:
                past_key_values = DynamicCompressFp8Cache.from_legacy_cache(past_key_values)
            else:
                past_key_values = DynamicCompressCache.from_legacy_cache(past_key_values)
        elif use_quantize_kv and not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
        elif (
            not use_quantize_kv
            and not use_compresskv
            and not isinstance(past_key_values, DynamicNormalCache)
        ):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
    # IPEX-LLM OPT end

    # `cache_position` is required after transformers 4.38
    if cache_position is not None:
        kwargs = {"cache_position": cache_position}
    else:
        kwargs = {}

    return LlamaModel.forward(
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


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, LlamaAttention)


def pre_compute_inv_freq(module: torch.nn.Module):
    if module.__class__.__name__ == "LlamaLinearScalingRotaryEmbedding":
        if hasattr(module, "scaling_factor"):
            module.register_buffer("inv_freq_scaled", None, persistent=False)
            module.inv_freq_scaled = module.inv_freq / module.scaling_factor


def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    kv_seq_len = key_states.shape[-2]
    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if query_states.device.type == "xpu":
        import xe_addons
        if hasattr(self, "rotary_emb"):
            # transformers < 4.46
            if hasattr(self.rotary_emb, "inv_freq_scaled"):
                xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq_scaled, position_ids,
                                               query_states, key_states)
            else:
                xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                               query_states, key_states)
        else:
            # transformers >= 4.46
            cos, sin = position_embeddings
            make_cache_contiguous_inplaced(cos, sin)
            xe_addons.rotary_half_with_cache_inplaced(query_states, key_states, cos, sin)
    else:
        if position_embeddings is None:
            if isinstance(getattr(self.rotary_emb, "cos_cached", None), torch.Tensor):
                # transformers < 4.38
                cos, sin = self.rotary_emb(value_states, kv_seq_len)
            else:
                # transformers >= 4.38
                cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids)

    if past_key_value is not None:
        # [CompressKV]
        if isinstance(past_key_value, DynamicCompressCache):
            enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, q_len)
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
        attention_mask, q_len == key_states.size(2)
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

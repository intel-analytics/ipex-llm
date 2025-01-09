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
# https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/blob/main/modeling_minicpm.py
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
import warnings
from typing import Optional, Tuple, Union, List
import math
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.models.utils import use_quantize_kv_cache
from ipex_llm.transformers.models.utils import should_use_compresskv, should_use_fuse_rope
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.kv import DynamicNormalCache, DynamicFp8Cache, \
    DynamicCompressCache, DynamicCompressFp8Cache
from transformers.cache_utils import Cache


def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, "MiniCPMAttention")
    merge_qkv_base(module, "MiniCPMSdpaAttention")


def apply_residual_scale(module: torch.nn.Module):
    if module.__class__.__name__ == "MiniCPMDecoderLayer":
        scale = module.scale_depth / math.sqrt(module.num_hidden_layers)
        module.self_attn.o_proj.weight.data *= scale
        if module.self_attn.o_proj.bias is not None:
            module.self_attn.o_proj.bias.weight.data *= scale
        module.mlp.down_proj.weight.data *= scale
        if module.mlp.down_proj.bias is not None:
            module.mlp.down_proj.bias.weight.data *= scale


def minicpm_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_key_value_heads,
                                                        self.num_key_value_heads], dim=1)

    # [CompressKV]
    use_compresskv = isinstance(past_key_value, DynamicCompressCache)
    use_quantizekv = isinstance(past_key_value, DynamicFp8Cache)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        if self.rotary_emb.__class__.__name__ == "MiniCPMLongRoPE":
            if kv_seq_len > self.rotary_emb.original_max_position_embeddings:
                inv_freq = self.rotary_emb.long_inv_freq
            else:
                inv_freq = self.rotary_emb.short_inv_freq
        else:
            inv_freq = self.rotary_emb.inv_freq
        import xe_addons
        xe_addons.rotary_half_inplaced(inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, "llama"
        )

    if past_key_value is not None:
        if use_compresskv:
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
        attention_mask, q_len == kv_seq_len
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def minicpm_model_forward_wrapper(origin_forward):
    def minicpm_model_forward(
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
        # ipex-llm changes start
        # IPEX-LLM OPT: kv cache and quantize kv cache
        inputs = input_ids if input_ids is not None else inputs_embeds
        use_quantize_kv = use_quantize_kv_cache(self.layers[0].mlp.up_proj, inputs,
                                                self.config.num_attention_heads,
                                                self.config.num_key_value_heads)
        use_compress_kv = should_use_compresskv(inputs, inputs.shape[1]) or \
            isinstance(past_key_values, DynamicCompressCache)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache:
            if use_compress_kv and not isinstance(past_key_values,
                                                  DynamicCompressCache):
                if use_quantize_kv:
                    past_key_values = DynamicCompressFp8Cache.from_legacy_cache(past_key_values)
                else:
                    past_key_values = DynamicCompressCache.from_legacy_cache(past_key_values)
            elif (use_quantize_kv and not use_compress_kv
                  and not isinstance(past_key_values, DynamicFp8Cache)):
                past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
            elif (not use_quantize_kv and not use_compress_kv
                  and not isinstance(past_key_values, DynamicNormalCache)):
                past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
        # ipex-llm changes end
        return origin_forward(
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

    return minicpm_model_forward


def minicpm_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )

    # ipex-llm changes start
    hidden_states = residual + hidden_states
    # ipex-llm changes end

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    hidden_states = self.mlp(hidden_states)

    # ipex-llm changes start
    hidden_states = residual + hidden_states
    # ipex-llm changes end

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

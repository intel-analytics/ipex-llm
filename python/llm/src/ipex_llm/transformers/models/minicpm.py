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
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import math
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.models.utils import use_sdp, use_sdp_causal, use_quantize_kv_cache
from ipex_llm.transformers.models.utils import restore_fp8_kv_cache, get_compresskv_attn_mask
from ipex_llm.transformers.models.utils import should_use_compresskv, should_use_fuse_rope
from ipex_llm.transformers.models.llama import repeat_kv
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.kv import DynamicNormalCache, DynamicFp8Cache, \
    DynamicCompressCache, DynamicCompressFp8Cache
from transformers.cache_utils import Cache


def merge_qkv(module: torch.nn.Module):
    return merge_qkv_base(module, "MiniCPMAttention")


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
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
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
    if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import xe_addons
        # [CompressKV]
        if use_compresskv:
            attention_mask = get_compresskv_attn_mask(key_states, attention_mask)

        if use_quantizekv:
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                            attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                        attention_mask)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if use_quantizekv:
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states,
                                               value_states, attention_mask)
    else:
        if use_quantizekv:
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

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
                                                self.config.num_attention_heads //
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

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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
# which is licensed under Apache License 2.0:
#
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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

import torch

from typing import Optional, Tuple
from transformers.cache_utils import EncoderDecoderCache

from ipex_llm.transformers.utils import invalidInputError
from ipex_llm.transformers.models.common import scaled_dot_product_attention


def whisper_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[EncoderDecoderCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    invalidInputError(not output_attentions and layer_head_mask is None,
                      "`output_attentions` and `layer_head_mask` are not supported")

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

    if past_key_value is not None:
        is_updated = past_key_value.is_updated.get(self.layer_idx)
        if is_cross_attention:
            past_key_value.is_updated[self.layer_idx] = True
            past_key_value = past_key_value.cross_attention_cache
        else:
            past_key_value = past_key_value.self_attention_cache

    # use key_value_states if cross attention
    current_states = key_value_states if key_value_states is not None else hidden_states
    if is_cross_attention and past_key_value and is_updated:
        # reuse k,v, cross_attentions
        key_states = past_key_value.key_cache[self.layer_idx]
        value_states = past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self._shape(self.k_proj(current_states), -1, bsz)
        value_states = self._shape(self.v_proj(current_states), -1, bsz)
        if past_key_value is not None:
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

    # IPEX-LLM OPT: sdpa
    is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False
    attn_output = scaled_dot_product_attention(
        query_states,
        key_states.contiguous(),
        value_states.contiguous(),
        attention_mask,
        is_causal
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value

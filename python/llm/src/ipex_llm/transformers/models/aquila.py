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
# https://huggingface.co/BAAI/AquilaChat-7B/blob/main/modeling_aquila.py
#
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

import math
import torch

from typing import Optional, Tuple
from ipex_llm.transformers.models.common import attention_softmax
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb, should_use_fuse_rope
from ipex_llm.transformers.models.utils import update_past_key_value
from ipex_llm.utils.common import log4Error


def aquila_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)\
        .view(bsz, q_len, self.num_heads, self.head_dim)\
        .transpose(1, 2)
    key_states = self.k_proj(hidden_states)\
        .view(bsz, q_len, self.num_heads, self.head_dim)\
        .transpose(1, 2)
    value_states = self.v_proj(hidden_states)\
        .view(bsz, q_len, self.num_heads, self.head_dim)\
        .transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids, "aquila")

    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, False, hidden_states.device
    )

    past_key_value = (key_states, value_states) if use_cache else None

    attn_weights = torch.matmul(query_states,
                                key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    attn_weights = torch.clamp(attn_weights, min=-1024., max=1024.)
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        log4Error.invalidInputError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
            f"but is {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            log4Error.invalidInputError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                f"but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights,
            torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        )

    # upcast attention to fp32
    attn_weights = attention_softmax(attn_weights)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        log4Error.invalidInputError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

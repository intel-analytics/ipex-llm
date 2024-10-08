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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional


def mllama_vision_attention_forward(
    self,
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = None,
):
    query = self.q_proj(hidden_state)
    key = self.k_proj(hidden_state)
    value = self.v_proj(hidden_state)

    batch_size, q_seq_len, _ = query.shape
    _, kv_seq_len, _ = key.shape

    query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    from ipex_llm.transformers.models.common import attention_softmax
    attn_weights = attention_softmax(attn_weights, self.training)

    attn_output = torch.matmul(attn_weights, value)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

    output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return output, attn_weights

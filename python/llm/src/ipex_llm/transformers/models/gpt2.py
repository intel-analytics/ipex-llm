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

import torch
from ipex_llm.transformers.models.utils import use_sdp_non_causal


def gpt2_attention_attn(
    self,
    query,
    key,
    value,
    attention_mask=None,
    head_mask=None
):
    # ipex-llm changes start
    if (
        self.scale_attn_weights
        and not self.scale_attn_by_inverse_layer_idx
        and head_mask is None
        and query.size(-2) == key.size(-2)
        and use_sdp_non_causal(query.size(-1), query.device, query.dtype)
    ):
        if not self.is_cross_attention:
            seq_len = query.size(-2)
            causal_mask = self.bias[:, :, :seq_len, :seq_len]
            mask_value = torch.finfo(query.dtype).min
            mask_value = torch.full([], mask_value, dtype=query.dtype, device=query.device)
            attention_mask = attention_mask.expand(-1, -1, seq_len, seq_len)
            attention_mask = torch.where(causal_mask, attention_mask, mask_value)
        else:
            attention_mask = attention_mask.expand(-1, -1, seq_len, seq_len)

        import xe_addons
        attn_weights = None
        attn_output = xe_addons.sdp_non_causal(query, key.contiguous(),
                                               value.contiguous(), attention_mask)
        return attn_output, attn_weights
    # ipex-llm changes end

    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype,
                                device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights

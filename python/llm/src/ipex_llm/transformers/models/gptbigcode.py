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


from typing import Optional, Tuple, Union
import torch


def _attn_wrapper(origin_attn):
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_output, attn_weights = origin_attn(self,
                                                query=query.to(key.dtype),
                                                key=key,
                                                value=value,
                                                attention_mask=attention_mask,
                                                head_mask=head_mask)
        if query.device.type == 'xpu' and 1 < query.numel() // query.size(-1) <= 64:
            attn_output = attn_output.clone()
        return attn_output, attn_weights
    return _attn


def gptbigcode_attention_forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False):

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                from ipex_llm.utils.common import invalidInputError
                invalidInputError(
                    False,
                    "If class is used as cross attention," +
                    "the weights `q_attn` have to be defined. " +
                    "Please make sure to instantiate class with " +
                    "`GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split(
                (self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        if layer_past is not None:
            if layer_past.shape[-2] == key_value.shape[-2]:
                key_value = torch.cat((layer_past, key_value), dim=-2)
            else:
                fill_zeros = torch.zeros(layer_past.shape[0],
                                         layer_past.shape[1],
                                         key_value.shape[2] - layer_past.shape[2],
                                         dtype=layer_past.dtype,
                                         device=layer_past.device)
                layer_past = torch.cat([layer_past, fill_zeros], dim=-1)
                key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        attn_output, attn_weights = self._attn(query, key.transpose(-1, -2),
                                               value, attention_mask, head_mask)

        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)

        return outputs

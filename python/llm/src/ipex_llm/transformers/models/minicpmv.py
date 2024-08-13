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
from typing import Optional
from ipex_llm.transformers.models.common import merge_qkv_base
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor


def merge_qkv(module: torch.nn.Module):
    return merge_qkv_base(module, "SiglipAttention")


def siglip_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
):
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=1)

    attn_weights = torch.matmul(query_states * self.scale, key_states.transpose(2, 3))
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    import xe_addons
    xe_addons.attn_softmax_inplaced(attn_weights)

    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights


def patched_repetition_penalty_call(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    if scores.device.type == "xpu":
        import xe_addons
        xe_addons.repetition_penalty_logits_process_inplaced(scores, input_ids, self.penalty)
    else:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, input_ids, score)
    return scores


def minicpmv_generate_wrapper(origin_generate):
    def generate(
        *inputs,
        **kwargs
    ):
        RepetitionPenaltyLogitsProcessor.__call__ = patched_repetition_penalty_call
        return origin_generate(
            *inputs,
            **kwargs,
        )
    return generate

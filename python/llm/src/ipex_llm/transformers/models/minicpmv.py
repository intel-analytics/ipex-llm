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


import math
import torch
from typing import Optional, List
from torch.nn.functional import linear
from ipex_llm.transformers.models.common import merge_qkv_base
from ipex_llm.transformers.models.common import attention_softmax
from transformers import AutoProcessor
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor


# MiniCPM-V-2_5 and MiniCPM-V-2_6
def merge_qkv(module: torch.nn.Module):
    merge_qkv_base(module, "SiglipAttention")
    merge_qkv_base(module, "Idefics2VisionAttention")


# MiniCPM-V-2_5 and MiniCPM-V-2_6
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

    attn_weights = attention_softmax(attn_weights, self.training)

    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights


# MiniCPM-V-2_6
def _in_projection_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for
            # better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2)
            proj = proj.contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for
            # better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2)
            kv_proj = kv_proj.contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        # ipex-llm changes start: add contiguous to workaround a ipex bug
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        w_q = w_q.contiguous()
        w_k = w_k.contiguous()
        w_v = w_v.contiguous()
        # ipex-llm changes end
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# MiniCPM-V-2
# modified from timm.models.vision_transformer.Attention.forward
def vision_transformer_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    bsz, q_len, hidden_size = x.size()

    qkv = self.qkv(x)
    qkv = qkv.view(bsz, q_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=1)

    attn_weights = torch.matmul(query_states * self.scale, key_states.transpose(2, 3))
    attn_weights = attention_softmax(attn_weights, self.training)
    attn_weights = self.attn_drop(attn_weights)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)
    return attn_output


# MiniCPM-V-2_5
def minicpmv_chat_wrapper(origin_chat):
    def minicpmv_chat(
        self,
        image,
        msgs,
        tokenizer,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=1024,
        sampling=True,
        max_inp_length=2048,
        system_prompt='',
        stream=False,
        **kwargs
    ):
        if processor is None:
            if getattr(self, "processor", None) is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path,
                                                               trust_remote_code=True)
            processor = self.processor
        return origin_chat(
            self=self,
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            processor=processor,
            vision_hidden_states=vision_hidden_states,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            max_inp_length=max_inp_length,
            system_prompt=system_prompt,
            stream=stream,
            **kwargs
        )
    return minicpmv_chat


# MiniCPM-V-2
def minicpmv_get_vision_embedding(self, pixel_values):
    res = []
    dtype = self.dtype

    def process_each_pixel(pixel_value, dtype, config, vpm, resampler):
        H, W = pixel_value.shape[-2:]
        target_size = (math.ceil(H / config.patch_size), math.ceil(W / config.patch_size))
        vision_embedding = self.vpm_forward_features(pixel_value.unsqueeze(0).type(dtype))

        if hasattr(vpm, 'num_prefix_tokens') and vpm.num_prefix_tokens > 0:
            vision_embedding = vision_embedding[:, vpm.num_prefix_tokens:]
        return resampler(vision_embedding, target_size)

    for pixel_value in pixel_values:
        result = process_each_pixel(pixel_value, dtype, self.config, self.vpm, self.resampler)
        res.append(result)
    return torch.vstack(res)


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

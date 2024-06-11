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
# https://huggingface.co/internlm/internlm-chat-7b/blob/659ed911eec1e26810f9854f19c5ec27854e9cf3/modeling_internlm.py
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
""" PyTorch InternLM model."""
import math
from typing import Optional, Tuple, List

import torch
import torch.utils.checkpoint
from torch import nn
from ipex_llm.transformers.models.utils import should_use_fuse_rope, apply_rotary_pos_emb
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb_cache_freq_xpu
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, restore_fp8_kv_cache
from ipex_llm.transformers.models.utils import update_past_key_value
from ipex_llm.transformers.models.utils import use_sdp, use_sdp_causal
from einops import rearrange


def merge_qkv(module: torch.nn.Module):
    if module.__class__.__name__ == "InternLMAttention":
        new_weight = torch.cat([
            module.q_proj.weight.data,
            module.k_proj.weight.data,
            module.v_proj.weight.data,
        ], dim=0)
        new_bias = torch.cat([
            module.q_proj.bias.data,
            module.k_proj.bias.data,
            module.v_proj.bias.data,
        ], dim=-1)

        qkv_proj = torch.nn.Linear(0, 0, bias=True)
        qkv_proj.weight = torch.nn.Parameter(new_weight, requires_grad=False)
        qkv_proj.bias = torch.nn.Parameter(new_bias, requires_grad=False)
        qkv_proj.in_features = new_weight.size(1)
        qkv_proj.out_features = new_weight.size(0)
        module.qkv_proj = qkv_proj

        del module.q_proj, module.k_proj, module.v_proj


def internlm_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # IPEX-LLM OPT: fuse rope
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, "internlm"
        )

    # IPEX-LLM OPT: kv cache and quantzie kv cache
    use_quantize_kv = use_quantize_kv_cache(self.qkv_proj, hidden_states)
    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, use_quantize_kv, hidden_states.device
    )
    past_key_value = (key_states, value_states) if use_cache else None

    # IPEX-LLM OPT: sdp
    attn_weights = None
    if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                            attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states, attention_mask)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states,
                                               value_states, attention_mask)
    else:
        if use_quantize_kv:
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def internlm2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)
    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # IPEX-LLM OPT: fuse rope
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, "internlm"
        )

    # IPEX-LLM OPT: kv cache and quantzie kv cache
    use_quantize_kv = use_quantize_kv_cache(self.wqkv, hidden_states)
    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, use_quantize_kv, hidden_states.device
    )
    past_key_value = (key_states, value_states) if use_cache else None

    # IPEX-LLM OPT: sdp
    attn_weights = None
    if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                            attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states, attention_mask)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states,
                                               value_states, attention_mask)
    else:
        if use_quantize_kv:
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def pre_process_attn_and_mlp(module: torch.nn.Module):
    if module.__class__.__name__ == "InternLM2Attention":
        module.wqkv_lora_scaling = module.wqkv.lora_scaling
        module.wqkv_Plora_A = module.wqkv.Plora_A
        module.wqkv_Plora_B = module.wqkv.Plora_B
        del module.wqkv.Plora_A
        del module.wqkv.Plora_B

        module.wo_lora_scaling = module.wo.lora_scaling
        module.wo_Plora_A = module.wo.Plora_A
        module.wo_Plora_B = module.wo.Plora_B
        del module.wo.Plora_A
        del module.wo.Plora_B

    elif module.__class__.__name__ == "InternLM2MLP":
        module.w1_lora_scaling = module.w1.lora_scaling
        module.w1_Plora_A = module.w1.Plora_A
        module.w1_Plora_B = module.w1.Plora_B
        del module.w1.Plora_A
        del module.w1.Plora_B

        module.w2_lora_scaling = module.w2.lora_scaling
        module.w2_Plora_A = module.w2.Plora_A
        module.w2_Plora_B = module.w2.Plora_B
        del module.w2.Plora_A
        del module.w2.Plora_B

        module.w3_lora_scaling = module.w3.lora_scaling
        module.w3_Plora_A = module.w3.Plora_A
        module.w3_Plora_B = module.w3.Plora_B
        del module.w3.Plora_A
        del module.w3.Plora_B


def add_lora(x: torch.Tensor, result: torch.Tensor,
             im_mask: torch.Tensor = None, lora_scaling: float = 0,
             Plora_A: torch.nn.Linear = None, Plora_B: torch.nn.Linear = None):
    if im_mask is not None and torch.sum(im_mask) > 0:
        part_x = x[im_mask]
        result[im_mask] += Plora_B(Plora_A(part_x) * lora_scaling)
    return result


def internlm_xcomposser2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    im_mask: Optional[Tuple[torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    qkv_states = self.wqkv(hidden_states)
    qkv_states = add_lora(hidden_states, qkv_states, im_mask, self.wqkv_lora_scaling,
                          self.wqkv_Plora_A, self.wqkv_Plora_B)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # IPEX-LLM OPT: fuse rope
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(
            query_states, key_states, sin, cos, "internlm", position_ids
        )
    else:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, "internlm")

    # IPEX-LLM OPT: kv cache and quantzie kv cache
    use_quantize_kv = use_quantize_kv_cache(self.wqkv, hidden_states)
    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, use_quantize_kv, device
    )
    past_key_value = (key_states, value_states) if use_cache else None

    # IPEX-LLM OPT: sdp
    if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                            attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states, attention_mask)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states,
                                               value_states, attention_mask)
    else:
        if use_quantize_kv:
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output_2 = self.wo(attn_output)

    attn_output = add_lora(attn_output, attn_output_2, im_mask, self.wo_lora_scaling,
                           self.wo_Plora_A, self.wo_Plora_B)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def internlm_xcomposser2_mlp_forward(
    self,
    x: torch.Tensor,
    im_mask: Optional[Tuple[torch.Tensor]] = None,
):
    w1 = self.w1(x)
    w1 = add_lora(x, w1, im_mask, self.w1_lora_scaling, self.w1_Plora_A, self.w1_Plora_B)
    w3 = self.w3(x)
    w3 = add_lora(x, w3, im_mask, self.w3_lora_scaling, self.w3_Plora_A, self.w3_Plora_B)
    x = self.act_fn(w1) * w3
    w2 = self.w2(x)
    w2 = add_lora(x, w2, im_mask, self.w2_lora_scaling, self.w2_Plora_A, self.w2_Plora_B)
    return w2


@torch.no_grad()
def internlm_xcomposser2_chat(
    self,
    tokenizer,
    query: str,
    image: torch.Tensor = None,
    history: List[Tuple[str, str]]=[],
    streamer=None,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.8,
    repetition_penalty: float=1.005,
    meta_instruction:
    str = ('You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
           '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model'
           'that is developed by Shanghai AI Laboratory (上海人工智能实验室).'
           'It is designed to be helpful, honest, and harmless.\n'
           '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the'
           'language chosen by the user such as English and 中文.\n'
           '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating'
           'responses effectively based on the provided image.'),
    **kwargs,
):
    if image is None:
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        im_mask = torch.zeros(inputs['input_ids'].shape[:2]).bool()
    else:
        image = self.encode_img(image)
        inputs, im_mask = self.interleav_wrap_chat(tokenizer, query, image,
                                                   history, meta_instruction)
    inputs = {
        k: v.to(device=self.device, dtype=self.dtype)
        for k, v in inputs.items() if torch.is_tensor(v)
    }
    im_mask = im_mask.to(self.device)
    # also add end-of-assistant token in eos token id to avoid unnecessary generation
    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
    ]
    outputs = self.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
        im_mask=im_mask,
        **kwargs,
    )
    if image is None:
        outputs = outputs[0].cpu().tolist()[len(inputs['input_ids'][0]):]
    else:
        outputs = outputs[0].cpu().tolist()
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    response = response.split('[UNUSED_TOKEN_145]')[0]
    history = history + [(query, response)]
    return response, history

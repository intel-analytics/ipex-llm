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
# https://huggingface.co/Qwen/Qwen-7B-Chat/blob/be72f02dd47087f9035ee9bb5dea571b84785d27/modeling_qwen.py
#
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from typing import Optional, Tuple, Union, Callable, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.utils import logging
from ipex_llm.transformers.models.utils import update_past_key_value, should_use_fuse_rope
from ipex_llm.transformers.models.utils import restore_fp8_kv_cache, use_quantize_kv_cache
from ipex_llm.transformers.models.utils import rotate_half, SILU
from ipex_llm.transformers.models.utils import mlp_fusion_check
from ipex_llm.transformers.models.utils import use_flash_attention, use_sdp, use_sdp_causal
from ipex_llm.utils.common import invalidInputError, invalidOperationError
from transformers.modeling_outputs import BaseModelOutputWithPast


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = (t_ * cos) + (rotate_half(t_) * sin)
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


def qwen_attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb_list: Optional[List[torch.Tensor]] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    invalidInputError(not self.use_flash_attn and not self.use_cache_quantization,
                      "flash attn and kv_cache quantization are not supported")
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    past_key_value = (None if layer_past is None
                      else (layer_past[0].transpose(1, 2), layer_past[1].transpose(1, 2)))

    qkv = self.c_attn(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    kv_seq_len = key_states.shape[2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    # IPEX-LLM OPT: fuse rope
    position_ids = rotary_pos_emb_list[-1]  # the last one is posisiton_ids
    inv_freq = rotary_pos_emb_list[-2]
    rotary_pos_emb_list = rotary_pos_emb_list[:-2]
    invalidInputError(len(rotary_pos_emb_list) == 1,
                      "rotary_pos_emb_list's length cannot be larger than 1")
    use_fuse_rope = should_use_fuse_rope(hidden_states, position_ids, self.training)
    rotary_pos_emb = rotary_pos_emb_list[0]
    if use_fuse_rope:
        rot_dim = rotary_pos_emb[0].size(-1)
        import xe_addons
        xe_addons.rotary_half_inplaced(inv_freq, position_ids,
                                       query_states[..., :rot_dim],
                                       key_states[..., :rot_dim])
    else:
        rotary_pos_emb = [i[:, -q_len:, :, :].transpose(1, 2) for i in rotary_pos_emb]
        query_states = apply_rotary_pos_emb(query_states, rotary_pos_emb)
        key_states = apply_rotary_pos_emb(key_states, rotary_pos_emb)

    if kv_seq_len > self.seq_length and self.use_logn_attn and not self.training:
        seq_start = kv_seq_len - q_len
        seq_end = kv_seq_len
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].transpose(1, 2)
        query_states = query_states * logn_tensor.type_as(query_states).expand_as(query_states)

    # IPEX-LLM OPT: kv cache and quantzie kv cache
    use_quantize_kv = use_quantize_kv_cache(self.c_attn, hidden_states)
    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, use_quantize_kv, device
    )
    past_key_value = (key_states.transpose(1, 2),
                      value_states.transpose(1, 2)) if use_cache else None

    # IPEX-LLM OPT: sdp
    attn_weights = None
    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, attention_mask):
        attn_output = F.scaled_dot_product_attention(query_states.to(dtype=torch.float16),
                                                     key_states.to(dtype=torch.float16),
                                                     value_states.to(dtype=torch.float16),
                                                     is_causal=True).to(hidden_states.dtype)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states, value_states, None)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states, value_states, None)
    else:
        if q_len > 1:
            causal_mask = torch.tril(
                torch.ones((kv_seq_len, kv_seq_len), dtype=torch.bool, device=query_states.device)
            ).view(1, 1, kv_seq_len, kv_seq_len)
            causal_mask = causal_mask[
                :, :, kv_seq_len - q_len:kv_seq_len, :kv_seq_len
            ]
            attention_mask = torch.zeros(causal_mask.shape, dtype=query_states.dtype,
                                         device=query_states.device)
            attention_mask.masked_fill_(causal_mask.logical_not(),
                                        torch.finfo(attention_mask.dtype).min)
            attention_mask = attention_mask.expand([bsz, -1, -1, -1])
        else:
            attention_mask = None

        if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
            import xe_addons
            if use_quantize_kv:
                attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                                attention_mask)
            else:
                attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                            attention_mask)
        else:
            if use_quantize_kv:
                key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                                query_states.dtype)
            attn_weights = torch.matmul(query_states,
                                        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            if self.softmax_in_fp32:
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                           dtype=torch.float32).to(
                                                               value_states.dtype)
            else:
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.c_proj(attn_output)

    if output_attentions:
        return attn_output, past_key_value, attn_weights
    else:
        return attn_output, past_key_value


def qwen_attention_forward_registered(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb_list: Optional[List[torch.Tensor]] = None,
    registered_causal_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # invalidInputError(not self.use_flash_attn and not self.use_cache_quantization,
    #                   "flash attn and kv_cache quantization are not supported")
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    past_key_value = (None if layer_past is None
                      else (layer_past[0].transpose(1, 2), layer_past[1].transpose(1, 2)))

    qkv = self.c_attn(hidden_states)
    qkv = qkv.view(bsz, q_len, self.num_heads * 3, self.head_dim)
    qkv = qkv.transpose(1, 2)
    query_states, key_states, value_states = qkv.split([self.num_heads,
                                                        self.num_heads,
                                                        self.num_heads], dim=1)

    kv_seq_len = key_states.shape[2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    # IPEX-LLM OPT: fuse rope
    position_ids = rotary_pos_emb_list[-1]  # the last one is posisiton_ids
    inv_freq = rotary_pos_emb_list[-2]
    rotary_pos_emb_list = rotary_pos_emb_list[:-2]
    invalidInputError(len(rotary_pos_emb_list) == 1,
                      "rotary_pos_emb_list's length cannot be larger than 1")
    use_fuse_rope = should_use_fuse_rope(hidden_states, position_ids, self.training)
    rotary_pos_emb = rotary_pos_emb_list[0]
    if use_fuse_rope:
        rot_dim = rotary_pos_emb[0].size(-1)
        import xe_addons
        xe_addons.rotary_half_inplaced(inv_freq, position_ids,
                                       query_states[..., :rot_dim], key_states[..., :rot_dim])
    else:
        rotary_pos_emb = [i[:, -q_len:, :, :].transpose(1, 2) for i in rotary_pos_emb]
        query_states = apply_rotary_pos_emb(query_states, rotary_pos_emb)
        key_states = apply_rotary_pos_emb(key_states, rotary_pos_emb)

    if kv_seq_len > self.seq_length and self.use_logn_attn and not self.training:
        seq_start = kv_seq_len - q_len
        seq_end = kv_seq_len
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].transpose(1, 2)
        query_states = query_states * logn_tensor.type_as(query_states).expand_as(query_states)

    # IPEX-LLM OPT: kv cache and quantzie kv cache
    use_quantize_kv = use_quantize_kv_cache(self.c_attn, hidden_states)
    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, use_quantize_kv, device
    )
    past_key_value = (key_states.transpose(1, 2),
                      value_states.transpose(1, 2)) if use_cache else None

    # IPEX-LLM OPT: sdp
    attn_weights = None
    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, attention_mask):
        attn_output = F.scaled_dot_product_attention(query_states.to(dtype=torch.float16),
                                                     key_states.to(dtype=torch.float16),
                                                     value_states.to(dtype=torch.float16),
                                                     is_causal=True).to(hidden_states.dtype)
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim, query_states, self.training):
        import xe_addons
        if use_quantize_kv:
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states, value_states, None)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states, value_states, None)
    else:
        if q_len > 1:
            causal_mask = registered_causal_mask[
                :, :, kv_seq_len - q_len:kv_seq_len, :kv_seq_len
            ]
            attention_mask = torch.zeros(causal_mask.shape, dtype=query_states.dtype,
                                         device=query_states.device)
            attention_mask.masked_fill_(causal_mask.logical_not(),
                                        torch.finfo(attention_mask.dtype).min)
            attention_mask = attention_mask.expand([bsz, -1, -1, -1])
        else:
            attention_mask = None

        if use_sdp(q_len, kv_seq_len, self.head_dim, query_states):
            import xe_addons
            if use_quantize_kv:
                attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                                attention_mask)
            else:
                attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                            attention_mask)
        else:
            if use_quantize_kv:
                key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                                query_states.dtype)
            attn_weights = torch.matmul(query_states,
                                        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            if self.softmax_in_fp32:
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                           dtype=torch.float32).to(
                                                               value_states.dtype)
            else:
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.c_proj(attn_output)

    if output_attentions:
        return attn_output, past_key_value, attn_weights
    else:
        return attn_output, past_key_value


def qwen_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    qtype = getattr(self.w1, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training) and not self.w1.enable_xetla:
        import xe_linear
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        return self.c_proj(xe_linear.mlp_forward_xpu(
            x_2d, self.w2.weight.data, self.w1.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.w2.out_len,
            SILU, qtype
        ))
    return self.c_proj(F.silu(self.w2(x)) * self.w1(x))


def qwen_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(
            False,
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        invalidInputError(False, "You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        if self.use_cache_quantization:
            past_length = past_key_values[0][0][0].size(2)
        else:
            past_length = past_key_values[0][0].size(1)
    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    if attention_mask is not None:
        if batch_size <= 0:
            invalidInputError(False, "batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    hidden_states = inputs_embeds

    kv_seq_len = hidden_states.size()[1]
    if past_key_values[0] is not None:
        # past key values[0][0] shape: bs * seq_len * head_num * dim
        if self.use_cache_quantization:
            kv_seq_len += past_key_values[0][0][0].shape[2]
        else:
            kv_seq_len += past_key_values[0][0].shape[1]

    if self.training or not self.use_dynamic_ntk:
        ntk_alpha_list = [1.0]
    elif kv_seq_len != hidden_states.size()[1]:
        ntk_alpha_list = self.rotary_emb._ntk_alpha_cached_list
    else:
        ntk_alpha_list = []
        if attention_mask is not None and kv_seq_len > self.seq_length:
            true_seq_lens = attention_mask.squeeze(1).squeeze(1).eq(0).sum(dim=-1,
                                                                           dtype=torch.int32)
            for i in range(hidden_states.size()[0]):
                true_seq_len = true_seq_lens[i].item()
                ntk_alpha = self.get_ntk_alpha(true_seq_len)
                ntk_alpha_list.append(ntk_alpha)
        else:
            ntk_alpha = self.get_ntk_alpha(kv_seq_len)
            ntk_alpha_list.append(ntk_alpha)
    self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
    # ipex-llm changes
    rotary_pos_emb_list = [
        self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
    ] + [self.rotary_emb.inv_freq.to(self.dtype), position_ids]
    # ipex-llm changes ends

    hidden_states = self.drop(hidden_states)
    output_shape = input_shape + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                rotary_pos_emb_list,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            # ipex-llm changes
            curr_device = block.ln_1.weight.device
            from accelerate.utils.operations import send_to_device
            if rotary_pos_emb_list is not None:
                rotary_pos_emb_list = send_to_device(rotary_pos_emb_list, curr_device)
            if attention_mask is not None:
                attention_mask = send_to_device(attention_mask, curr_device)
            if head_mask[i] is not None:
                head_mask[i] = send_to_device(head_mask[i], curr_device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = send_to_device(encoder_hidden_states, curr_device)
            if encoder_attention_mask is not None:
                encoder_attention_mask = send_to_device(encoder_attention_mask,
                                                        curr_device)
            # ipex-llm changes ends

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb_list=rotary_pos_emb_list,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, presents, all_hidden_states] if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )

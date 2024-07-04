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
# https://huggingface.co/Qwen/Qwen-VL-Chat/blob/bbe5a805de49a41b7343d240ab84d4c305caa265/modeling_qwen.py
#
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import math
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.utils import logging
from ipex_llm.transformers.models.utils import extend_kv_cache, init_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import rotate_half
from ipex_llm.transformers.models.utils import use_sdp
from transformers.modeling_outputs import BaseModelOutputWithPast
from ipex_llm.utils.common import invalidInputError
import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    rot_dim = freqs[0].shape[-1]
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = (t_ * cos) + (rotate_half(t_) * sin)
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        # Due to issue https://github.com/intel/intel-extension-for-pytorch/issues/454,
        # currently put interpolation execution into cpu
        return F.interpolate(
            abs_pos.to("cpu").float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype).to(abs_pos.device)
    else:
        return abs_pos


def qwen_attention_forward_vl(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb: Optional[List[torch.Tensor]] = None,
    registered_causal_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):

    kv_seq_len = hidden_states.size()[1]

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    use_fuse_rope = should_use_fuse_rope(self, hidden_states)
    mixed_x_layer = self.c_attn(hidden_states)
    query, key, value = mixed_x_layer.split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)
    if rotary_pos_emb is not None:
        cur_len = query.shape[1]
        rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
        rotary_pos_emb = (rotary_pos_emb,) * 2
        q_pos_emb, k_pos_emb = rotary_pos_emb
        # Slice the pos emb for current inference
        query = apply_rotary_pos_emb(query, q_pos_emb)
        key = apply_rotary_pos_emb(key, k_pos_emb)
    query_size, key_size = query.size(1), key.size(1)

    if layer_past is not None:
        kv_seq_len += layer_past[0].shape[1]
        # past_key, past_value = layer_past[0], layer_past[1]
        # key = torch.cat((past_key, key), dim=1)
        # value = torch.cat((past_value, value), dim=1)
        cache_k = layer_past[0].transpose(1, 2)
        cache_v = layer_past[1].transpose(1, 2)
        if cache_k.stride()[1] < kv_seq_len * cache_k.size(3):
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=hidden_states.device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v

        key_states, value_states = append_kv_cache(cache_k, cache_v,
                                                   key.transpose(1, 2), value.transpose(1, 2))
        key = key_states
        value = value_states
    elif use_cache:
        max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
        new_key_states, new_value_states = init_kv_cache(bsz,
                                                         self.num_heads,
                                                         self.head_dim,
                                                         kv_seq_len,
                                                         max_cache_length,
                                                         dtype=key.dtype,
                                                         device=hidden_states.device)
        new_key_states[:] = key.transpose(1, 2)
        new_value_states[:] = value.transpose(1, 2)
        key = new_key_states
        value = new_value_states

    if use_cache:
        present = (key.transpose(1, 2), value.transpose(1, 2))
    else:
        present = None

    if self.use_logn_attn and not self.training:
        if self.logn_tensor.device != query.device or self.logn_tensor.dtype != query.dtype:
            self.logn_tensor = self.logn_tensor.to(query.device).type_as(query)
        seq_start = key_size - key_size
        seq_end = key_size
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
        query = query * logn_tensor.expand_as(query)

    query = query.permute(0, 2, 1, 3)

    if not self.training and not hidden_states.requires_grad and \
            use_sdp(q_len, key.shape[2], self.head_dim, query):
        import xe_addons
        attn_output = xe_addons.sdp(query, key, value, attention_mask)
        attn_output = attn_output.view(query.shape)
        attn_output = attn_output.transpose(1, 2)
        attn_weight = None
    else:
        attn_output, attn_weight = self._attn(
            query, key, value, registered_causal_mask, attention_mask, head_mask
        )

    context_layer = self._merge_heads(
        attn_output, self.num_heads, self.head_dim
    )

    attn_output = self.c_proj(context_layer)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weight,)

    return outputs


def should_use_fuse_rope(self, query_states):
    use_fuse_rope = query_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and query_states.requires_grad)
    return use_fuse_rope


def is_enough_kv_cache_room(layer_past, kv_seq_len=1):
    # to determinate if is enough kv cache room in transformers between 4.31 and 4.35
    # seq_len for current seq len
    # For llama like kv cache, i.e., [bs, n_head, seq_len, head_dim]
    if layer_past is None:
        return False
    else:
        cache_k, cache_v = layer_past[0], layer_past[1]
        cache_k = cache_k.transpose(1, 2)
        return cache_k.stride(1) < (kv_seq_len + 1) * cache_k.size(3)


def qwen_vl_resampler_forward(self, x, attn_mask=None):

    pos_embed = get_abs_pos(self.pos_embed, x.size(1))

    x = self.kv_proj(x)
    x = self.ln_kv(x).permute(1, 0, 2)

    N = x.shape[1]
    q = self.ln_q(self.query)
    out = self.attn(
        self._repeat(q, N) + self.pos_embed.unsqueeze(1),
        x + pos_embed.unsqueeze(1),
        x,
        attn_mask=attn_mask)[0]
    return out.permute(1, 0, 2)


def qwen_vl_vision_transformer_forward(self, x: torch.Tensor):
    x = x.to(
        dtype=self.transformer.get_cast_dtype(),
        device=self.transformer.get_cast_device(),
    )
    # to patches
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    x = x + get_abs_pos(self.positional_embedding, x.size(1))

    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.attn_pool(x)
    x = self.ln_post(x)
    x = x @ self.proj

    return x


def qwen_vl_model_forward(
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
    # bigdl-llm change starts
    input = input_ids if input_ids is not None else inputs_embeds
    # bigdl-llm change ends
    if past_key_values is None and torch.any(input == self.config.visual['image_start_id']):
        bos_pos = torch.where(input == self.config.visual['image_start_id'])
        eos_pos = torch.where(input == self.config.visual['image_start_id'] + 1)
        invalidInputError((bos_pos[0] == eos_pos[0]).all(),
                          'bos_pos[0] should be same as eos_pos[0]')
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input[i][a + 1: b - 1].tolist()
            image = image[: image.index(self.config.visual['image_start_id'] + 2)]
            images.append(bytes(image).decode('utf-8'))

        images = self.visual.encode(images)
        invalidInputError(images.shape[0] == len(images),
                          'images.shape[0] should be same as len(images)')
        fake_images = None
    elif self.training:
        fake_images = torch.zeros(1, 3, 224, 224).to(
            dtype=self.visual.conv1.weight.dtype, device=self.visual.conv1.weight.device)
        images = self.visual(fake_images)
    else:
        fake_images = None
        images = None

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
        invalidInputError(False,
                          "You cannot specify both input_ids and inputs_embeds at the same time")
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
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if batch_size <= 0:
        invalidInputError(False, "batch_size has to be defined and > 0")
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_length
    )

    hidden_states = inputs_embeds

    kv_seq_len = hidden_states.size()[1]
    if past_key_values[0] is not None:
        # past key values[0][0] shape: bs * seq_len * head_num * dim
        kv_seq_len += past_key_values[0][0].shape[1]
    if (
        self.use_dynamic_ntk
        and kv_seq_len == hidden_states.size()[1]
        and not self.training
    ):
        context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
    else:
        ntk_alpha = self.rotary_emb._ntk_alpha_cached

    rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
    for idx in range(len(rotary_pos_emb)):
        rotary_pos_emb[idx] = rotary_pos_emb[idx].to(hidden_states.device)

    hidden_states = self.drop(hidden_states).clone()
    if fake_images is not None:
        hidden_states = hidden_states + images.mean()*0
    elif images is not None:
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[i][a + 1: b] = images[idx]
    output_shape = input_shape + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
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
                rotary_pos_emb,
                self.registered_causal_mask,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb=rotary_pos_emb,
                registered_causal_mask=self.registered_causal_mask,
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

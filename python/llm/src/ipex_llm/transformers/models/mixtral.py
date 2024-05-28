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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py

# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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

""" PyTorch Mixtral model."""
import math
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)

import torch
from torch import nn
import torch.nn.functional as F
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb,\
    apply_rotary_pos_emb_cache_freq_xpu, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.models.mistral import should_use_fuse_rope
from ipex_llm.transformers.models.utils import use_decoding_fast_path
from ipex_llm.transformers.models.utils import use_flash_attention, use_sdp
from ipex_llm.transformers.models.utils import mlp_fusion_check, SILU
from ipex_llm.transformers.low_bit_linear import IQ2_XXS

import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def mixtral_moeblock_forward(self,
                             hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    bs = hidden_states.shape[0]
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    if bs == 1:
        selected_experts = selected_experts[0].cpu().tolist()
        for idx in range(self.top_k):
            exp_id = selected_experts[idx]
            expert_layer = self.experts[exp_id]
            weight = routing_weights[:, idx]
            if idx == 0:
                final_hidden_states = expert_layer(hidden_states, weight)
            else:
                final_hidden_states = final_hidden_states + expert_layer(hidden_states, weight)
    elif bs < 256 and hidden_states.device.type == 'xpu':
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim),
                                          dtype=hidden_states.dtype, device=hidden_states.device)
        import xe_linear
        indexes = xe_linear.get_moe_indexes(selected_experts.to(torch.int32).cpu(), 8)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx_list = indexes[0][expert_idx]
            top_x_list = indexes[1][expert_idx]
            if len(idx_list) == 0:
                continue

            top_x = torch.tensor(top_x_list, dtype=torch.long, device=hidden_states.device)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state,
                                                 routing_weights[top_x_list, idx_list, None])
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    else:
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts,
                                                  num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state,
                                                 routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def mixtral_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    padding_mask: Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx)
    decoding_fast_path = use_decoding_fast_path(self.q_proj,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len)

    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k = past_key_value.key_cache[self.layer_idx]
        cache_v = past_key_value.value_cache[self.layer_idx]
        kv_seq_len = cache_k.shape[-2]
        import xe_linear
        query_states, key_states, value_states = xe_linear.forward_qkv(hidden_states,
                                                                       self.q_proj.weight,
                                                                       self.k_proj.weight,
                                                                       self.v_proj.weight,
                                                                       position_ids,
                                                                       cache_k, cache_v,
                                                                       self.q_proj.weight.qtype,
                                                                       self.v_proj.weight.qtype,
                                                                       kv_seq_len,
                                                                       self.head_dim,
                                                                       self.rotary_emb.base,)
        kv_seq_len += 1
        # update past_key_value's seem_tokens and kv caches.
        if self.layer_idx == 0:
            past_key_value.seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states
    # diasble it for now as it will cause output change for unknown reason
    # elif decoding_fast_path and self.q_proj.qtype == IQ2_XXS:
    #     # this path self.v_proj use q4_0
    #     hidden_states = hidden_states.view(1, -1)
    #     cache_k = past_key_value.key_cache[self.layer_idx]
    #     cache_v = past_key_value.value_cache[self.layer_idx]
    #     kv_seq_len = cache_k.shape[-2]
    #     import xe_linear
    #     query_states, key_states = xe_linear.forward_qk(hidden_states,
    #                                                       self.q_proj.weight,
    #                                                       self.k_proj.weight,
    #                                                       position_ids,
    #                                                       cache_k,
    #                                                       self.q_proj.weight.qtype,
    #                                                       kv_seq_len,
    #                                                       self.head_dim,
    #                                                       10000)
    #     kv_seq_len += 1
    #     # update past_key_value's seem_tokens and kv caches.
    #     if self.layer_idx == 0:
    #         past_key_value.seen_tokens = kv_seq_len
    #     # update value_states
    #     value_states = self.v_proj(hidden_states)
    #     value_states = value_states.view(bsz, q_len,
    #                                      self.num_key_value_heads, self.head_dim).transpose(1, 2)
    #     new_size = (cache_v.size(0),
    #                 cache_v.size(1),
    #                 cache_v.size(2) + value_states.size(2),
    #                 cache_v.size(3))
    #     new_cache_v = cache_v.as_strided(new_size, cache_v.stride(), storage_offset=0)
    #     new_cache_v[:, :, cache_v.size(2):cache_v.size(2)+value_states.size(2), :] = value_states

    #     past_key_value.key_cache[self.layer_idx] = key_states
    #     past_key_value.value_cache[self.layer_idx] = new_cache_v
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                invalidInputError(False,
                                  "The cache structure has changed since version v4.36. "
                                  f"If you are using {self.__class__.__name__} for "
                                  "auto-regressive decodingwith k/v caching, please make sure "
                                  "to initialize the attention class with a layer index.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if use_fuse_rope:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            query_states, key_states = apply_rotary_pos_emb_cache_freq_xpu(query_states,
                                                                           key_states,
                                                                           sin,
                                                                           cos,
                                                                           "mixtral")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "mixtral")

        if past_key_value is not None:
            # update the number of seen tokens
            if self.layer_idx == 0:
                past_key_value.seen_tokens += key_states.shape[-2]

            # reuse k, v, self_attention
            # update `past_key_value` with `key_states` and `value_states` for layer `layer_idx`
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(key_states)
                past_key_value.value_cache.append(value_states)
            else:
                cache_k = past_key_value.key_cache[self.layer_idx]
                cache_v = past_key_value.value_cache[self.layer_idx]

                if not enough_kv_room:
                    # allocate new
                    new_c_k, new_c_v = extend_kv_cache(bsz,
                                                       self.num_key_value_heads,  # Support GQA
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=device)

                    new_c_k[:] = cache_k
                    new_c_v[:] = cache_v
                    cache_k = new_c_k
                    cache_v = new_c_v

                key_states, value_states = append_kv_cache(cache_k,
                                                           cache_v,
                                                           key_states,
                                                           value_states)

                # update past_key_value
                past_key_value.key_cache[self.layer_idx] = key_states
                past_key_value.value_cache[self.layer_idx] = value_states

    if not self.training and not hidden_states.requires_grad:
        fsdp_flag = use_flash_attention(query_states, key_states)
    else:
        fsdp_flag = False
    if fsdp_flag:
        attention_dtype = torch.float16  # use fp16 for flash attention
    else:
        attention_dtype = original_dtype

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups).to(device,
                                                                     dtype=attention_dtype)
    value_states = repeat_kv(value_states, self.num_key_value_groups).to(device,
                                                                         dtype=attention_dtype)

    if fsdp_flag:
        attn_output = F.scaled_dot_product_attention(query_states.to(dtype=attention_dtype),
                                                     key_states,
                                                     value_states,
                                                     is_causal=True)
        attn_weights = None
    elif use_sdp(query_states.shape[2], key_states.shape[2], self.head_dim, query_states):
        import xe_addons
        attn_output = xe_addons.sdp(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        attn_weights = torch.matmul(
            query_states.to(key_states.dtype),
            key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            invalidInputError(
                False,
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)},"
                f" but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},"
                    f" but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.\
            softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(
            False,
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)},"
            f" but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def mixtral_mlp_forward(
    self,
    x: torch.Tensor,
    routing_weights
) -> torch.Tensor:
    qtype = getattr(self.w1, "qtype", None)
    if mlp_fusion_check(x, qtype, self.training) and not self.w1.enable_xetla:
        import xe_linear
        return self.w2(xe_linear.mlp_forward_xpu(
            x, self.w1.weight.data, self.w3.weight.data,
            x.shape[0], x.shape[1], self.w1.out_len,
            SILU, qtype,
        )) * routing_weights
    else:
        current_hidden_states = self.act_fn(self.w1(x)) * self.w3(x)
        current_hidden_states = self.w2(current_hidden_states)
        return routing_weights * current_hidden_states


def mixtral_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, MoeModelOutputWithPast]:
    # to be compatible with transformers>=4.37.0
    self._use_flash_attention_2 = self.config._attn_implementation == "flash_attention_2"

    output_attentions = output_attentions if output_attentions is not None \
        else self.config.output_attentions
    output_router_logits = (
        output_router_logits if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False, "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")  # noqa
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        invalidInputError(False, "You have to specify either decoder_input_ids or decoder_inputs_embeds")  # noqa

    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is not None and self._use_flash_attention_2 and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            invalidInputError(
                False,
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "  # noqa
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask \
            if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."  # noqa
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        else:
            # bigdl-llm changes:
            #
            # Avoid moving `attention_mask`` and `position_ids`` to other devices multiple times.
            #
            # When the model is partitioned on two different devices using
            # `accelerate`'s `dispatch``, a hook to move inputs to the correct device is
            # added to each layer's `forward``, which will result in moving `attention_mask`
            # and `position_ids`, which allocated on device:0, to other devices for each
            # decoder layer not in device:0.
            #
            # To avoid this, we move `attention_mask` and `position_ids` to the device of
            # the current layer before the forward call, so that the moving is only done once
            # for each devices other than devie:0.
            #
            curr_device = decoder_layer.input_layernorm.weight.device
            if attention_mask is not None:
                attention_mask = attention_mask.to(curr_device)
            if position_ids is not None:
                position_ids = position_ids.to(curr_device)
            # bigdl-llm changes end
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() \
            if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]  # noqa
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )

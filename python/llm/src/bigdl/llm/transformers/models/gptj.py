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
# This file is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
#

import torch
from typing import Optional, Tuple, Union
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, \
    apply_rotary_pos_emb, append_kv_cache, apply_ipex_rotate_every_two
from transformers.utils.import_utils import is_torch_fx_proxy
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gptj.modeling_gptj import GPTJModel
from bigdl.llm.utils.common import invalidInputError


KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def _get_embed_positions(self, position_ids):
    embed_positions = self.embed_positions
    if embed_positions.device != position_ids.device:
        embed_positions = embed_positions.to(position_ids.device)
        self.embed_positions = embed_positions
    return embed_positions.repeat(position_ids.shape[0], 1, 1)


def _attn(
    self,
    query,
    key,
    value,
    attention_mask=None,
    head_mask=None,
):
    # compute causal mask from causal mask buffer
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

    # Keep the attention weights computation in fp32 to avoid overflow issues
    query = query.to(torch.float32)
    key = key.to(torch.float32)

    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    mask_value = torch.finfo(attn_weights.dtype).min
    # Need to be a tensor, otherwise we get error:
    # `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    attn_weights = attn_weights / self.scale_attn

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def gptj_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    rotary_emb: Optional[Tuple]=None,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Tuple[torch.Tensor]],
    Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
]:
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
    value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

    sin, cos = rotary_emb
    use_fuse_rope = hidden_states.device.type == "xpu" and not self.training

    if self.rotary_dim is not None:
        k_rot = key[:, :, :, : self.rotary_dim]
        q_rot = query[:, :, :, : self.rotary_dim]

        if use_fuse_rope:
            apply_ipex_rotate_every_two(q_rot, k_rot, cos, sin)
        else:
            k_pass = key[:, :, :, self.rotary_dim:]
            q_pass = query[:, :, :, self.rotary_dim:]
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids, "gptj")
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
    else:
        if use_fuse_rope:
            apply_ipex_rotate_every_two(query, key, cos, sin)
        else:
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids, "gptj")

    batch_size, q_len, _ = hidden_states.size()

    key = key.permute(0, 2, 1, 3).contiguous()
    query = query.permute(0, 2, 1, 3).contiguous()

    kv_seq_len = key.size(-2)
    device = hidden_states.device

    if layer_past is not None:
        kv_seq_len += layer_past[0].size(2)

    if layer_past is not None:
        cache_k = layer_past[0]
        cache_v = layer_past[1]
        past_length = cache_k.size(2)
        if cache_k.stride()[1] < kv_seq_len * cache_k.size(3):
            new_cache_k, new_cache_v = extend_kv_cache(batch_size,
                                                       self.num_attention_heads,
                                                       self.head_dim,
                                                       past_length,
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_v.dtype,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v
        key, value = append_kv_cache(cache_k, cache_v, key, value)

    elif use_cache:
        key_cache, value_cache = init_kv_cache(batch_size,
                                               self.num_attention_heads,
                                               self.head_dim,
                                               kv_seq_len,
                                               kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                               dtype=value.dtype,
                                               device=device)
        key_cache[:] = key
        value_cache[:] = value
        key = key_cache
        value = value_cache

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    # compute self-attention: V x Softmax(QK^T)
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


def gptj_block_forward(
    self,
    hidden_states: Optional[torch.FloatTensor],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    rotary_emb: Optional[Tuple]=None,
    output_attentions: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states=hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        use_cache=use_cache,
        rotary_emb=rotary_emb,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]

    feed_forward_hidden_states = self.mlp(hidden_states)
    hidden_states = attn_output + feed_forward_hidden_states + residual

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions)


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j",
                                torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


old_init = GPTJModel.__init__


def gptj_model_new_init(self, config):
    old_init(self, config)
    embed_dim = config.hidden_size
    rotary_dim = config.rotary_dim
    pos_embd_dim = rotary_dim or embed_dim
    max_positions = config.max_position_embeddings
    self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)


def get_new_embed_positions(position_ids, prev_embed_positions):
    embed_positions = prev_embed_positions
    if embed_positions.device != position_ids.device:
        embed_positions = embed_positions.to(position_ids.device)
        prev_embed_positions = embed_positions
    return embed_positions.repeat(position_ids.shape[0], 1, 1), prev_embed_positions


def gptj_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None \
        else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False,
                          "You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
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

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length,
                                    dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

    # Attention mask.
    if attention_mask is not None:
        if batch_size <= 0:
            invalidInputError(False, "batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x num_attention_heads x N x N
    # head_mask has shape n_layer x batch x num_attention_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    hidden_states = inputs_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing."
                "Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    # Repeat cos sin here, call only once for each token.
    # If put this to attension forward, it will generate too many times.
    if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
        # The logic to conditionally copy to GPU could not be traced, so we do this
        # every time in the torch.fx case
        embed_positions = get_embed_positions(self.embed_positions, position_ids)
    else:
        embed_positions, self.embed_positions = get_new_embed_positions(position_ids,
                                                                        self.embed_positions)

    repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                None,
                attention_mask,
                position_ids,
                head_mask[i],
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                rotary_emb=(sin, cos),
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                     if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )

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
# https://huggingface.co/THUDM/chatglm2-6b/blob/8eb45c842594b8473f291d0f94e7bbe86ffc67d8/modeling_chatglm.py
#

import torch
from typing import Optional, Tuple, List
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.llama import get_ipex_version


KV_CACHE_ALLOC_BLOCK_LENGTH = 256
KV_CACHE_ALLOC_MIN_LENGTH = 512


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


@torch.jit.script
def apply_rotary_pos_emb_chatglm(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


def chatglm_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu" and not (self.training and hidden_states.requires_grad):
        if get_ipex_version() <= "2.0.110+xpu":
            hidden_states, _ = torch.ops.torch_ipex.rms_norm(hidden_states,
                                                             [self.weight.size(0)], self.weight)
        else:
            # for ipex >= 2.1
            hidden_states = torch.ops.torch_ipex.fast_rms_norm(hidden_states,
                                                               [self.weight.size(0)],
                                                               self.weight,
                                                               None,  # bias
                                                               self.eps)
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
    return hidden_states


def chatglm2_model_forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        full_attention_mask: Optional[torch.BoolTensor]=None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        use_cache: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (
                past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids,
                                                 past_key_values,
                                                 padding_mask=attention_mask)

    use_fuse_rope = input_ids.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not self.training

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    if use_fuse_rope:
        # Repeat cos sin here, call only once for each token.
        # Chatglm2's rotary embedding is similar to gptj's, is rotate_every_two.
        # If put this to attension forward, it will generate too many times.
        cos, sin = rotary_pos_emb.split(rotary_pos_emb.shape[-1] // 2, dim=-1)
        cos = cos.squeeze(-1)
        sin = sin.squeeze(-1)
        cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        rotary_pos_emb = (cos, sin)
    else:
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                     if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def chatglm2_attention_forward_8eb45c(
        self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================

    # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
    device = hidden_states.device
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition,
                                       self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition,
                                     self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition,
                                                        3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    cur_length, batch_size = query_layer.shape[0], query_layer.shape[1]

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        if len(rotary_pos_emb) == 2:  # use_fuse_rope, see chatglm2_model_forward
            cos, sin = rotary_pos_emb
            rot_dim = cos.shape[-1]
            query_layer = query_layer.transpose(0, 1)
            key_layer = key_layer.transpose(0, 1)
            query_layer_cur = query_layer[..., :rot_dim]
            key_layer_cur = key_layer[..., :rot_dim]
            # ipex's apply_rotary_embedding can change the origin storage, so query_layer will get
            # the result directly.
            torch.ops.torch_ipex.apply_rotary_embedding(query_layer_cur, sin, cos, query_layer_cur)
            torch.ops.torch_ipex.apply_rotary_embedding(key_layer_cur, sin, cos, key_layer_cur)
            query_layer = query_layer.transpose(0, 1)
            key_layer = key_layer.transpose(0, 1)
        else:
            query_layer = apply_rotary_pos_emb_chatglm(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb_chatglm(key_layer, rotary_pos_emb)

    if self.multi_query_attention:
        key_length = key_layer.size(0)
        query_group_size = self.num_attention_heads_per_partition // \
            self.num_multi_query_groups_per_partition
        key_layer = key_layer.permute(1, 2, 0, 3).unsqueeze(-3)  # [bs, nh/k, sl, hn]
        key_layer = key_layer.expand(-1, -1, query_group_size, -1, -1)
        key_layer = key_layer.contiguous().view((batch_size,
                                                 self.num_attention_heads_per_partition,
                                                 key_length,
                                                 self.hidden_size_per_attention_head))
        value_layer = value_layer.permute(1, 2, 0, 3).unsqueeze(-3)
        value_layer = value_layer.expand(-1, -1, query_group_size, -1, -1)
        value_layer = value_layer.contiguous().view((batch_size,
                                                     self.num_attention_heads_per_partition,
                                                     key_length,
                                                     self.hidden_size_per_attention_head))

    # adjust key and value for inference
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        cache_k = cache_k.permute(1, 2, 0, 3)
        cache_v = cache_v.permute(1, 2, 0, 3)
        past_length = cache_k.size(2)

        if cache_k.stride()[1] <= cache_k.size(2) * cache_k.size(3):
            max_cache_length = past_length + cur_length + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_cache_k, new_cache_v = extend_kv_cache(batch_size,
                                                       self.num_attention_heads_per_partition,
                                                       self.hidden_size_per_attention_head,
                                                       past_length,
                                                       max_cache_length,
                                                       dtype=query_layer.dtype,
                                                       device=device)
            new_cache_k[:] = cache_k
            new_cache_v[:] = cache_v
            cache_k = new_cache_k
            cache_v = new_cache_v

        key_layer, value_layer = append_kv_cache(cache_k, cache_v, key_layer, value_layer)

    elif use_cache:

        max_cache_length = max(KV_CACHE_ALLOC_MIN_LENGTH, cur_length) \
            + KV_CACHE_ALLOC_BLOCK_LENGTH
        key_cache, value_cache = init_kv_cache(batch_size, self.num_attention_heads_per_partition,
                                               self.hidden_size_per_attention_head, cur_length,
                                               max_cache_length,
                                               dtype=query_layer.dtype, device=device)
        key_cache[:] = key_layer
        value_cache[:] = value_layer
        key_layer = key_cache
        value_layer = value_cache

    if use_cache:
        kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    # ==================================
    # core attention computation
    # ==================================
    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

    # =================
    # Output. [sq, b, h]
    # =================

    output = self.dense(context_layer)

    return output, (key_layer.permute(2, 0, 1, 3), value_layer.permute(2, 0, 1, 3))


def core_attn_forward_8eb45c(self, query_layer, key_layer, value_layer, attention_mask):
    pytorch_major_version = int(torch.__version__.split('.')[0])
    if pytorch_major_version >= 2 and (query_layer.device.type == 'xpu' or query_layer.size(0) > 1):
        query_layer = query_layer.permute(1, 2, 0, 3)
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer,
                                                                             key_layer,
                                                                             value_layer,
                                                                             attention_mask,
                                                                             is_causal=True)
        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer,
                                                                             key_layer,
                                                                             value_layer,
                                                                             attention_mask)
        context_layer = context_layer.permute(2, 0, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
    else:
        # Raw attention scores

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2),
                       query_layer.size(0), key_layer.size(2))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1],
            output_size[2], output_size[3], dtype=query_layer.dtype,
            device=query_layer.device
        )

        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2], output_size[3], dtype=query_layer.dtype,
            device=query_layer.device
        )

        # Raw attention scores. [b * np, sq, sk]
        torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
            out=matmul_result
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(0), value_layer.size(1),
                       query_layer.size(0), value_layer.size(3))
        # change view [sk, b * np, hn]
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.empty(
            output_size[0] * output_size[1],
            output_size[2], value_layer.size(-1), dtype=value_layer.dtype,
            device=value_layer.device,
        )
        torch.bmm(attention_probs, value_layer, out=context_layer)
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer

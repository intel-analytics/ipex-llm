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

import os
import torch
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import update_past_key_value
from ipex_llm.transformers.models.utils import use_quantize_kv_cache
from ipex_llm.transformers.models.utils import should_use_fuse_rope, apply_rotary_pos_emb
from ipex_llm.transformers.models.utils import mlp_fusion_check, SILU
from ipex_llm.transformers.models.utils import use_quantize_kv_cache
from ipex_llm.transformers.models.utils import should_use_compresskv, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.kv import DynamicCompressCache, DynamicCompressFp8Cache

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


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

    if inputs_embeds is None:
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.embedding(input_ids)
    else:
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        seq_length, batch_size, _ = inputs_embeds.shape
        input_ids = torch.empty((batch_size, seq_length),
                                dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    if use_cache:
        use_compress_kv = should_use_compresskv(input_ids, input_ids.shape[1])
        n_heads = self.config.num_attention_heads
        if self.config.multi_query_attention:
            n_kv_heads = self.config.multi_query_group_num
        else:
            n_kv_heads = n_heads
        use_quantize_kv = use_quantize_kv_cache(self.encoder.layers[0].mlp.gate_proj,
                                                input_ids, n_heads, n_kv_heads)
        if use_compress_kv and not isinstance(past_key_values,
                                              DynamicCompressCache):
            if use_quantize_kv:
                past_key_values = DynamicCompressFp8Cache.from_legacy_cache(past_key_values)
            else:
                past_key_values = DynamicCompressCache.from_legacy_cache(past_key_values)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (
                past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids,
                                                 past_key_values,
                                                 padding_mask=attention_mask)

    # ipex-llm changes begin
    # 1. replace `rotary_pos_emb` with `inv_freq` and `position_ids`
    # 2. generate `causal_mask` and replace `full_attention_mask` with it
    if position_ids is None:
        if past_key_values is None:
            position_ids = torch.arange(seq_length, dtype=torch.int64, device=inputs_embeds.device)
        else:
            if isinstance(past_key_values, DynamicCompressCache):
                kv_length = past_key_values.get_seq_length()
            else:
                kv_length = past_key_values[0][0].size(0)
            position_ids = torch.arange(kv_length, kv_length + seq_length,
                                        dtype=torch.int64, device=inputs_embeds.device)
        position_ids = position_ids.repeat(batch_size, 1)

    if not getattr(self.rotary_pos_emb, "cached", False):
        rot_dim = self.rotary_pos_emb.dim
        base = 10000 * getattr(self.rotary_pos_emb, "rope_ratio", 1)
        inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2,
                                                dtype=torch.float,
                                                device=inputs_embeds.device) / rot_dim))
        inv_freq = inv_freq.to(inputs_embeds.dtype)
        self.rotary_pos_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rotary_pos_emb.cached = True

    # `full_attention_mask` is not None only when
    #  `past_key_values` is not None and `seq_length` > 1
    if full_attention_mask is not None:
        causal_mask = torch.zeros([batch_size, 1, seq_length, full_attention_mask.size(-1)],
                                  dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask.masked_fill_(full_attention_mask, mask_value)
    elif self.training or (inputs_embeds.device.type != "xpu" and past_key_values is None):
        full_attention_mask = self.get_masks(input_ids,
                                             past_key_values,
                                             padding_mask=attention_mask)
        causal_mask = torch.zeros([batch_size, 1, seq_length, full_attention_mask.size(-1)],
                                  dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask.masked_fill_(full_attention_mask, mask_value)
    else:
        causal_mask = None

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, causal_mask,
        rotary_pos_emb=(self.rotary_pos_emb.inv_freq, position_ids),
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )
    # ipex-llm changes end

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                     if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


# remove code which stores first token's kv cache by tensor format
# to fix chatglm2-32k and chatglm3-128k
def chatglm2_encoder_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
    use_cache: Optional[bool] = True,
    output_hidden_states: Optional[bool] = False,
):
    # [CompressKV]
    use_compress_kv = isinstance(kv_caches, DynamicCompressCache)

    if not kv_caches and not use_compress_kv:
        kv_caches = [None for _ in range(self.num_layers)]
    presents = () if use_cache else None
    if hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training:
        use_cache = False

    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None
    for index in range(self.num_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer = self._get_layer(index)
        if hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing \
                and self.training:
            layer_ret = torch.utils.checkpoint.checkpoint(
                layer,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_caches[index],
                use_cache
            )
        else:
            layer_ret = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches if use_compress_kv else kv_caches[index],
                use_cache=use_cache
            )
        hidden_states, kv_cache = layer_ret
        if use_cache:
            if use_compress_kv:
                presents = kv_caches
            else:
                presents = presents + (kv_cache,)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # Final layer norm.
    if self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states, presents, all_hidden_states, all_self_attentions


def chatglm2_attention_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    # hidden_states: [seq_len, bsz, head_dim]
    q_len, bsz, _ = hidden_states.size()

    # [CompressKV]
    use_compresskv = isinstance(kv_cache, DynamicCompressCache)

    # kv_cache: [seq_len, bsz, n_kv_head, head_dim] ->
    # past_key_value: [bsz, n_kv_head, seq_len, head_dim]
    if use_compresskv:
        past_key_value = kv_cache
    else:
        past_key_value = None if kv_cache is None else (kv_cache[0].permute(1, 2, 0, 3),
                                                        kv_cache[1].permute(1, 2, 0, 3))

    n_head = self.num_attention_heads_per_partition
    n_kv_head = self.num_multi_query_groups_per_partition if self.multi_query_attention else n_head
    head_dim = self.hidden_size_per_attention_head

    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(q_len, bsz, n_head + 2 * n_kv_head, head_dim)
    # [seq_len, bsz, n_head, head_dim] -> [bsz, n_head, seq_len, head_dim]
    qkv = qkv.permute(1, 2, 0, 3)

    query_states, key_states, value_states = qkv.split([n_head,
                                                        n_kv_head,
                                                        n_kv_head], dim=1)

    kv_seq_len = key_states.shape[2]
    if past_key_value is not None:
        if use_compresskv:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                           self.layer_number - 1)
        else:
            kv_seq_len += past_key_value[0].shape[2]

    # IPEX-LLM OPT: fuse rope
    inv_freq, position_ids = rotary_pos_emb
    rot_dim = inv_freq.size(-1) * 2
    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_two_inplaced(inv_freq, position_ids,
                                      query_states[..., :rot_dim], key_states[..., :rot_dim])
    else:
        idx_theta = torch.outer(position_ids[0].float(),
                                inv_freq.float()).to(hidden_states.dtype)
        idx_theta = idx_theta.unsqueeze(0).unsqueeze(0)
        cos = torch.cos(idx_theta).repeat_interleave(2, -1)
        sin = torch.sin(idx_theta).repeat_interleave(2, -1)
        q_rot, k_rot = apply_rotary_pos_emb(query_states[..., :rot_dim], key_states[..., :rot_dim],
                                            cos, sin, position_ids, "chatglm")
        query_states[..., :rot_dim] = q_rot[...]
        key_states[..., :rot_dim] = k_rot[...]

    # IPEX-LLM OPT: kv cache and quantize kv
    # [CompressKV]
    if use_compresskv:
        from transformers.configuration_utils import PretrainedConfig
        self.config = self.config if hasattr(self, "config") else PretrainedConfig()
        enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value,
                                                      self.layer_number - 1,
                                                      q_len)
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_number - 1,
            query_states, attention_mask, n_head // n_kv_head,
            self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH
        )
    else:
        use_quantize_kv = use_quantize_kv_cache(self.query_key_value, query_states,
                                                n_head, n_kv_head)
        key_states, value_states = update_past_key_value(
            past_key_value, key_states, value_states,
            kv_seq_len, use_quantize_kv, hidden_states.device
        )
        # past_key_value: [bsz, n_kv_head, seq_len, head_dim] -> [seq_len, bsz, n_kv_head, head_dim]
        past_key_value = (key_states.permute(2, 0, 1, 3),
                          value_states.permute(2, 0, 1, 3)) if use_cache else None

    # IPEX-LLM OPT: sdp
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == kv_seq_len
    )

    # context_layer's shape: [bsz, n_head, seq_len, head_dim] -> [seq_len, bsz, n_head * head_dim]
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(q_len, bsz, n_head * head_dim)
    output = self.dense(attn_output)

    return output, past_key_value


import torch.nn.functional as F


def split_mlp(module: torch.nn.Module):
    if module.__class__.__name__ == "MLP":
        gate_weight, up_weight = module.dense_h_to_4h.weight.data.chunk(2, dim=0)

        gate_proj = torch.nn.Linear(0, 0, bias=False)
        gate_proj.weight = torch.nn.Parameter(gate_weight, requires_grad=False)
        gate_proj.in_features = gate_weight.size(1)
        gate_proj.out_features = gate_weight.size(0)

        up_proj = torch.nn.Linear(0, 0, bias=False)
        up_proj.weight = torch.nn.Parameter(up_weight, requires_grad=False)
        up_proj.in_features = up_weight.size(1)
        up_proj.out_features = up_weight.size(0)

        module.gate_proj = gate_proj
        module.up_proj = up_proj

        module.activation_fn = F.silu

        del module.dense_h_to_4h


def mlp_forward(
    self,
    hidden_states: torch.FloatTensor
) -> torch.FloatTensor:
    x_2d = hidden_states.view(-1, hidden_states.shape[-1])
    qtype = getattr(self.gate_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training):
        x_2d = x_2d.contiguous()
        import xe_linear
        return self.dense_4h_to_h(xe_linear.mlp_forward_xpu(
            x_2d, self.gate_proj.weight.data, self.up_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.gate_proj.out_features,
            SILU, qtype
        ))
    return self.dense_4h_to_h(
        self.activation_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
    )

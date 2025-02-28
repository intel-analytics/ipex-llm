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
# https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py
# which is licensed under Apache License 2.0:
#
# https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE
#

import torch
import warnings

from typing import Optional, Tuple, List, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ipex_llm.utils.common.log4Error import invalidInputError
from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.transformers.models.common import padding_mla_v_hd_base
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import rotate_half


def padding_mla_v_hd(module: torch.nn.Module):
    padding_mla_v_hd_base(module, "DeepseekV3Attention")


def deepseek_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    invalidInputError((input_ids is None) ^ (inputs_embeds is None),
                      "You cannot specify both input_ids and inputs_embeds at the same time, "
                      "and must specify either one")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]

    # IPEX-LLM OPT start: kv cache
    past_key_values_length = 0
    use_cache = True if inputs_embeds.device.type == "xpu" else use_cache
    if use_cache:
        if not isinstance(past_key_values, DynamicNormalCache):
            past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)
    # IPEX-LLM OPT end: kv cache

    if position_ids is None:
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        position_ids = position_ids.unsqueeze(0)

    # IPEX-LLM OPT start: fuse rope
    if inputs_embeds.device.type == "xpu" and position_ids is not None:
        cos, sin = self.layers[0].self_attn.rotary_emb(inputs_embeds,
                                                       seq_length + past_key_values_length)
        cos = cos[position_ids[0]].contiguous()
        sin = sin[position_ids[0]].contiguous()
        position_embeddings = (cos, sin)
    else:
        position_embeddings = None
    # IPEX-LLM OPT end: fuse rope

    # 4d mask is passed through the layers
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def deepseek_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.q_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.q_head_dim], dim=-1
    )
    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is not None:
        query_states = q
        key_states = torch.cat(
            [k_nope, k_pe.expand([-1, self.num_heads, -1, -1])],
            dim=-1
        )
        import xe_addons
        cos, sin = position_embeddings
        xe_addons.rotary_two_with_cache_inplaced(query_states[:, :, :, self.qk_nope_head_dim:],
                                                 key_states[:, :, :, self.qk_nope_head_dim:],
                                                 cos, sin, True)
    else:
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, None)

    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == kv_seq_len, self.softmax_scale
    )
    attn_output = attn_output[:, :, :, :self.v_head_dim]

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def fuse_gate_forward(self, x: torch.Tensor):
    if x.device.type == "xpu" and x.dtype in [torch.float, torch.half]:
        x = x.view(-1, x.size(-1))
        logits = torch.nn.functional.linear(
            x.type(torch.float32), self.weight.type(torch.float32), None
        )
        scores = logits.sigmoid()

        import xe_addons
        topk_idx, topk_weight = xe_addons.moe_group_topk(
            scores, self.e_score_correction_bias,
            self.n_group, 2, self.topk_group, self.top_k,
            self.top_k > 1 and self.norm_topk_prob, 1e-20, self.routed_scaling_factor
        )
    else:
        topk_idx, topk_weight = self(x)
    return topk_idx, topk_weight.to(x.dtype)


def moe_infer_decode(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor):
    if (
        x.device.type == "xpu"
        and x.dtype in [torch.float, torch.half]
        and self.experts[0].down_proj.qtype == 2
    ):
        if getattr(self, "gates", None) is None:
            gate_addrs = [expert.gate_proj.weight.data_ptr() for expert in self.experts]
            up_addrs = [expert.up_proj.weight.data_ptr() for expert in self.experts]
            down_addrs = [expert.down_proj.weight.data_ptr() for expert in self.experts]
            gates = torch.tensor(gate_addrs, dtype=torch.uint64, device=x.device)
            ups = torch.tensor(up_addrs, dtype=torch.uint64, device=x.device)
            downs = torch.tensor(down_addrs, dtype=torch.uint64, device=x.device)
            self.register_buffer("gates", gates, persistent=False)
            self.register_buffer("ups", ups, persistent=False)
            self.register_buffer("downs", downs, persistent=False)

        import xe_linear
        final_out = xe_linear.moe_forward_vec(
            x, topk_ids, topk_weight, self.gates, self.ups, self.downs,
            x.size(-1), self.experts[0].intermediate_size, 2
        )
    else:
        idxs = topk_ids.flatten().tolist()
        outputs = []
        for i in idxs:
            expert = self.experts[i]
            expert_out = expert(x)
            outputs.append(expert_out)
        outs = torch.cat(outputs, dim=0)
        reshaped_topk_weight = topk_weight.squeeze(0).unsqueeze(-1)
        final_out = (outs * reshaped_topk_weight).sum(dim=0, keepdim=True)
    return final_out


def deepseek_moe_forward(self, hidden_states: torch.Tensor):
    identity = hidden_states
    orig_shape = hidden_states.shape
    # IPEX-LLM OPT start: fuse grouped topk in gate forward
    topk_idx, topk_weight = fuse_gate_forward(self.gate, hidden_states)
    # IPEX-LLM OPT end
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    flat_topk_idx = topk_idx.view(-1)
    if not self.training:
        # IPEX-LLM OPT start: add special moe_infer implementation for decoding
        if topk_idx.size(0) == 1 and self.ep_size == 1:
            y = moe_infer_decode(self, hidden_states, topk_idx, topk_weight)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        y = y.view(*orig_shape)
        # IPEX-LLM OPT end
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y

import torch
import warnings

from torch import nn
from typing import Optional, Tuple, List
from transformers.cache_utils import Cache

from ipex_llm.utils.common.log4Error import invalidInputError
from ipex_llm.transformers.models.utils import should_use_fuse_rope
from ipex_llm.transformers.models.utils import rotate_half
from ipex_llm.transformers.models.utils import use_sdp, use_sdp_causal
from ipex_llm.transformers.models.utils import use_quantize_kv_cache, restore_fp8_kv_cache
from ipex_llm.transformers.kv import DynamicNormalCache, DynamicFp8Cache


def pre_compute_inv_freq(module: torch.nn.Module):
    if module.__class__.__name__ == "MiniCPMLongRoPE":
        long_ext_factors = torch.tensor(module.long_factor, dtype=torch.float32)
        short_ext_factors = torch.tensor(module.short_factor, dtype=torch.float32)
        long_inv_freq = module.inv_freq * (1.0 / long_ext_factors)
        short_inv_freq = module.inv_freq * (1.0 / short_ext_factors)
        module.register_buffer("long_inv_freq", long_inv_freq, persistent=False)
        module.register_buffer("short_inv_freq", short_inv_freq, persistent=False)


def padding_v_head_dim(module: torch.nn.Module):
    if module.__class__.__name__ == "MiniCPMAttention":
        k_head_dim = module.qk_rope_head_dim + module.qk_nope_head_dim
        v_head_dim = module.v_head_dim
        invalidInputError(k_head_dim >= v_head_dim,
                          f"unsupported k_head_dim and v_head_dim: {k_head_dim} {v_head_dim}")
        if v_head_dim < k_head_dim:
            kv_b_proj = module.kv_b_proj
            w = kv_b_proj.weight.data.view(module.num_heads,
                                           module.qk_nope_head_dim + module.v_head_dim,
                                           module.kv_lora_rank)
            k_w, v_w = w.split([module.qk_nope_head_dim, module.v_head_dim], dim=1)
            new_v_w = torch.zeros([module.num_heads, k_head_dim, module.kv_lora_rank],
                                  dtype=v_w.dtype, device=v_w.device)
            new_v_w[:, :v_head_dim, :] = v_w
            new_w = torch.cat([k_w, new_v_w], dim=1).view(-1, module.kv_lora_rank)

            new_kv_b_proj = torch.nn.Linear(0, 0, bias=False,
                                            dtype=new_w.dtype, device=new_w.device)
            new_kv_b_proj.in_features = new_w.size(1)
            new_kv_b_proj.out_features = new_w.size(0)
            new_kv_b_proj.weight = torch.nn.Parameter(new_w, False)

            module.kv_b_proj = new_kv_b_proj


def minicpm3_model_forward_wrapper(origin_forward):
    def minicpm3_model_forward(
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
    ):
        # IPEX-LLM OPT: kv cache and quantize kv cache and sdp
        inputs = input_ids if input_ids is not None else inputs_embeds
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = True if inputs.device.type == "xpu" else use_cache
        use_quantize_kv = use_quantize_kv_cache(self.layers[0].mlp.down_proj, inputs)
        if use_cache:
            if use_quantize_kv and not isinstance(past_key_values, DynamicFp8Cache):
                past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
            elif not use_quantize_kv and not isinstance(past_key_values, DynamicNormalCache):
                past_key_values = DynamicNormalCache.from_legacy_cache(past_key_values)

        return origin_forward(
            self=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    return minicpm3_model_forward


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


def minicpm3_attention_forward(
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

    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

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

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        query_states = q
        key_states = torch.cat(
            [k_nope, k_pe.expand([-1, self.num_heads, -1, -1])],
            dim=-1
        )
        import xe_addons
        if self.rotary_emb.__class__.__name__ == "MiniCPMRotaryEmbedding":
            xe_addons.rotary_half_inplaced(inv_freq, position_ids,
                                           query_states[:, :, :, self.qk_nope_head_dim:],
                                           key_states[:, :, :, self.qk_nope_head_dim:])
        elif self.rotary_emb.__class__.__name__ == "MiniCPMLongRoPE":
            if kv_seq_len > self.rotary_emb.original_max_position_embeddings:
                inv_freq = self.rotary_emb.long_inv_freq
            else:
                inv_freq = self.rotary_emb.short_inv_freq
            xe_addons.rotary_half_inplaced(inv_freq, position_ids,
                                           query_states[:, :, :, self.qk_nope_head_dim:],
                                           key_states[:, :, :, self.qk_nope_head_dim:])
            if self.rotary_emb.scaling_factor != 1.0:
                query_states[:, :, :, self.qk_nope_head_dim:] *= self.rotary_emb.scaling_factor
                key_states[:, :, :, self.qk_nope_head_dim:] *= self.rotary_emb.scaling_factor
        else:
            invalidInputError(f"unknown rope method: {self.rotary_emb.__class__.__name__}")
    else:
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
    if use_sdp(q_len, kv_seq_len, self.q_head_dim, query_states):
        import xe_addons
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states,
                                            attention_mask)
        else:
            attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                        attention_mask)
        attn_output = attn_output[:, :, :, :self.v_head_dim]
    elif use_sdp_causal(q_len, kv_seq_len, self.q_head_dim, query_states, False):
        import xe_addons
        if isinstance(past_key_value, DynamicFp8Cache):
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        else:
            attn_output = xe_addons.sdp_causal(query_states, key_states,
                                               value_states, attention_mask)
        attn_output = attn_output[:, :, :, :self.v_head_dim]
    else:
        if isinstance(past_key_value, DynamicFp8Cache):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states[:, :, :, :self.v_head_dim])

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

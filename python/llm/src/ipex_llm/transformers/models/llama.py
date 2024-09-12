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
# https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/modeling_llama.py
# which is licensed under Apache License 2.0:
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import torch
import warnings
import importlib
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import math
import os
import torch.nn.functional as F
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import SILU
from ipex_llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache, should_use_compresskv, \
    get_compresskv_attn_mask
from ipex_llm.transformers.models.utils import is_enough_kv_cache_room_4_31, \
    apply_rotary_pos_emb, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu
from ipex_llm.transformers.models.utils import use_flash_attention, use_sdp, use_sdp_causal
from ipex_llm.transformers.models.utils import mlp_fusion_check, fp16_fusion_check
from ipex_llm.transformers.models.utils import use_decoding_fast_path, get_q_proj_or_qkv_proj
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention
from ipex_llm.transformers.low_bit_linear import SYM_INT4, FP8E5, IQ2_XXS, FP4
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.common import merge_qkv_base, fuse_mlp_base

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = Tuple[torch.Tensor]
from transformers import logging


logger = logging.get_logger(__name__)


def merge_qkv(module: torch.nn.Module):
    return merge_qkv_base(module, LlamaAttention)


def llama_decoding_fast_path_qtype_check(proj):
    # IQ2_XXS only can be used in Llama-like model
    qtype = getattr(proj, "qtype", None)
    return qtype in [SYM_INT4, FP8E5, IQ2_XXS, FP4]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states
    go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


_ipex_version = None


def get_ipex_version():

    global _ipex_version
    if _ipex_version is not None:
        return _ipex_version

    import intel_extension_for_pytorch as ipex
    _ipex_version = ipex.__version__
    return _ipex_version


def llama_model_forward_4_36(
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
    from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicCompressCache, \
        DynamicCompressFp8Cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    input = input_ids if input_ids is not None else inputs_embeds
    if use_cache:
        use_quantize = use_quantize_kv_cache(
            self.layers[0].mlp.up_proj, input,
            self.config.num_attention_heads//self.config.num_key_value_heads)
        use_compresskv = should_use_compresskv(input, input.shape[1]) or \
            isinstance(past_key_values, DynamicCompressCache)
        if use_compresskv:
            if not isinstance(past_key_values, DynamicCompressCache):
                if use_quantize:
                    past_key_values = DynamicCompressFp8Cache.from_legacy_cache(
                        past_key_values)
                else:
                    past_key_values = DynamicCompressCache.from_legacy_cache(
                        past_key_values)
        elif use_quantize:
            if not isinstance(past_key_values, DynamicFp8Cache):
                past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return llama_model_forward_4_36_internal(
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


def llama_model_forward_4_38(
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
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicCompressCache, \
        DynamicCompressFp8Cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    input = input_ids if input_ids is not None else inputs_embeds
    if use_cache:
        use_quantize = use_quantize_kv_cache(
            self.layers[0].mlp.up_proj, input,
            self.config.num_attention_heads//self.config.num_key_value_heads)
        use_compresskv = should_use_compresskv(input, input.shape[1]) or \
            isinstance(past_key_values, DynamicCompressCache)
        if use_compresskv:
            if not isinstance(past_key_values, DynamicCompressCache):
                if use_quantize:
                    past_key_values = DynamicCompressFp8Cache.from_legacy_cache(
                        past_key_values)
                else:
                    past_key_values = DynamicCompressCache.from_legacy_cache(
                        past_key_values)
        elif use_quantize:
            if not isinstance(past_key_values, DynamicFp8Cache):
                past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return llama_model_forward_4_38_internal(
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
        cache_position=cache_position,
    )


def llama_model_forward_4_41(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]]=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicCompressCache, \
        DynamicCompressFp8Cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    input = input_ids if input_ids is not None else inputs_embeds
    if use_cache:
        use_quantize = use_quantize_kv_cache(
            self.layers[0].mlp.up_proj, input,
            self.config.num_attention_heads//self.config.num_key_value_heads)
        use_compresskv = should_use_compresskv(input, input.shape[1]) or \
            isinstance(past_key_values, DynamicCompressCache)
        if use_compresskv:
            if not isinstance(past_key_values, DynamicCompressCache):
                if use_quantize:
                    past_key_values = DynamicCompressFp8Cache.from_legacy_cache(
                        past_key_values)
                else:
                    past_key_values = DynamicCompressCache.from_legacy_cache(
                        past_key_values)
        elif use_quantize:
            if not isinstance(past_key_values, DynamicFp8Cache):
                past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return llama_model_forward_4_41_internal(
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
        cache_position=cache_position,
    )


def llama_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu" and not (self.training and hidden_states.requires_grad):
        import xe_addons
        x_2d = hidden_states.reshape(-1, hidden_states.size(-1)).contiguous()
        output = xe_addons.rms_norm(self.weight, x_2d, self.variance_epsilon)
        return output.reshape(hidden_states.shape)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def llama_mlp_forward(
    self,
    x: torch.Tensor,
    residual=None
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    bsz, hidden_size = x_2d.shape
    qtype = getattr(self.gate_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training) and not self.down_proj.enable_xetla:
        import xe_linear
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out = self.down_proj(xe_linear.mlp_forward_xpu(
            x_2d, self.gate_proj.weight.data, self.up_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.gate_proj.out_len,
            SILU, qtype
        ))
        if residual is not None:
            return out + residual
        else:
            return out
    elif fp16_fusion_check(self.gate_proj, x, self.training) and \
            hidden_size == 4096 and bsz == 1:
        hidden_states1 = torch.ops.torch_ipex.mm_silu(x, self.gate_proj.weight)
        hidden_states = torch.ops.torch_ipex.mm_resmul(
            x, self.up_proj.weight, hidden_states1
        )
        if residual is None:
            hidden_states = torch.matmul(hidden_states, self.down_proj.weight)
        else:
            attn_output = torch.addmm(
                residual.flatten(0, -2),
                hidden_states.flatten(0, -2),
                self.down_proj.weight,
                beta=1,
            )
            hidden_states = attn_output.view(x.shape)
        return hidden_states
    elif x.device.type == "xpu" and not self.training:
        import xe_addons
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        xe_addons.mlp_silu_mul_inplaced(gate, up)
        out = self.down_proj(gate)
        if residual is not None:
            return out + residual
        else:
            return out
    else:
        a = self.act_fn(self.gate_proj(x))
        b = self.up_proj(x)
        c = a * b
        del a, b
        out = self.down_proj(c)
        if residual is not None:
            return out + residual
        else:
            return out


def should_use_fuse_rope(self, query_states, position_ids):
    use_fuse_rope = query_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and query_states.requires_grad)
    use_fuse_rope = use_fuse_rope and self.config.rope_scaling is None
    use_fuse_rope = use_fuse_rope and position_ids is not None
    return use_fuse_rope


# Only for xpu and training
def should_use_fast_rope(self, query_states, position_ids):
    use_fuse_rope = query_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and (self.training or query_states.requires_grad)
    use_fuse_rope = use_fuse_rope and self.config.rope_scaling is None
    use_fuse_rope = use_fuse_rope and position_ids is not None
    return use_fuse_rope


def should_split_qkv_tensor(query_states, bsz, num_heads, q_len, kv_seq_len, output_attentions):
    if not output_attentions:
        if os.environ.get("IPEX_LLM_SPLIT_QKV", None) is not None:
            return os.environ.get("IPEX_LLM_SPLIT_QKV", None) == "1"
        elif os.environ.get("IPEX_LLM_LOW_MEM", None) is not None:
            return os.environ.get("IPEX_LLM_LOW_MEM", None) == "1"
        elif query_states.dtype == torch.float16 and \
                query_states.shape[2] >= 6800:
            # split tensor for memory block limitation
            # support fp16 and set input length threshold at 6800 for now
            return True
        elif query_states.element_size()*bsz*num_heads*q_len*kv_seq_len >= 4*1024**3:
            # attn_weight size larger than memory block limitation 4GB
            return True
    return False


def llama_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37."
                "Please make sure use `attention_mask` instead.`"
            )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # add residual into mlp
    hidden_states = self.mlp(hidden_states, residual)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def fuse_qkv_weight_xetla(q_proj, k_proj, v_proj, qtype):
    if qtype == SYM_INT4:
        weight_size = q_proj.out_len * q_proj.in_len // 2
        zeros_size = q_proj.in_len * q_proj.out_len // 2 // 64
        zeros_end = weight_size + zeros_size
        weight_byte_shape = (q_proj.in_len//2, q_proj.out_len)
        zeros_byte_shape = (q_proj.in_len//64, q_proj.out_len//2)
        scales_byte_shape = (q_proj.in_len//64, q_proj.out_len*2)
        qweight = torch.concat([q_proj.weight.data[:weight_size].reshape(weight_byte_shape),
                                k_proj.weight.data[:weight_size].reshape(weight_byte_shape),
                                v_proj.weight.data[:weight_size].reshape(weight_byte_shape),
                                ], dim=-1).reshape(-1)
        qzeros = torch.concat([q_proj.weight.data[weight_size:zeros_end].reshape(zeros_byte_shape),
                               k_proj.weight.data[weight_size:zeros_end].reshape(zeros_byte_shape),
                               v_proj.weight.data[weight_size:zeros_end].reshape(zeros_byte_shape),
                               ], dim=-1).reshape(-1)
        qscales = torch.concat([q_proj.weight.data[zeros_end:].reshape(scales_byte_shape),
                                k_proj.weight.data[zeros_end:].reshape(scales_byte_shape),
                                v_proj.weight.data[zeros_end:].reshape(scales_byte_shape),
                                ], dim=-1).reshape(-1)
        q_proj.weight.data = torch.empty(0)
        k_proj.weight.data = torch.empty(0)
        v_proj.weight.data = torch.empty(0)
        return torch.cat([qweight, qzeros, qscales], dim=0)
    elif qtype == FP8E5:
        result = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=1).contiguous()
        q_proj.weight.data = torch.empty(0)
        k_proj.weight.data = torch.empty(0)
        v_proj.weight.data = torch.empty(0)
        return result
    else:
        invalidInputError(False, f"Unsupported qtype {qtype}")


def should_use_xetla_mm_qkv(self, device):
    if not hasattr(self, "q_proj"):
        # TODO: how to support xetla_mm_qkv for merged_qkv
        return False
    full_attn = self.q_proj.out_len == self.k_proj.out_len == self.v_proj.out_len
    supported_qtype = self.q_proj.qtype == SYM_INT4 and full_attn
    supported_qtype = supported_qtype or self.q_proj.qtype == FP8E5
    if self.q_proj.qtype == SYM_INT4 or self.q_proj.qtype == FP8E5:
        enable_xetla = self.q_proj.enable_xetla
    else:
        enable_xetla = False
    return device.type == "xpu" and enable_xetla and supported_qtype


def llama_attention_forward_4_31(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache(get_q_proj_or_qkv_proj(self), hidden_states,
                             self.num_key_value_groups):
        forward_function = llama_attention_forward_4_31_quantized
    else:
        forward_function = llama_attention_forward_4_31_original
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
        cache_position=cache_position,
        kwargs=kwargs
    )


def llama_attention_forward_4_31_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(getattr(self, "q_proj", None),
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp

    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        tmp_cache_k, tmp_cache_v = init_kv_cache(
            bsz,
            self.num_key_value_heads,
            self.head_dim,
            0,
            1,
            dtype=hidden_states.dtype,
            device=device
        )
        import xe_linear
        query_states, key_states, value_states = xe_linear.forward_qkv(hidden_states,
                                                                       self.q_proj.weight,
                                                                       self.k_proj.weight,
                                                                       self.v_proj.weight,
                                                                       position_ids,
                                                                       tmp_cache_k, tmp_cache_v,
                                                                       self.q_proj.weight.qtype,
                                                                       self.v_proj.weight.qtype,
                                                                       0,
                                                                       self.head_dim,
                                                                       self.rotary_emb.base,)
    else:
        if hasattr(self, "q_proj"):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            query_states, key_states, value_states = qkv.split([self.num_heads,
                                                                self.num_key_value_heads,
                                                                self.num_key_value_heads], dim=2)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if use_fuse_rope:
            import xe_addons
            xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                           query_states, key_states)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "llama")

    if past_key_value is None:
        kv_seq_len = key_states.shape[-2]
        repeated_key_states = repeat_kv(key_states, self.num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output, attn_weights = native_sdp(query_states, repeated_key_states,
                                               repeated_value_states,
                                               attention_mask, cache_position,
                                               bsz, q_len, kv_seq_len,
                                               self.head_dim, self.num_heads, output_attentions)
        if use_cache:
            k_cache, v_cache = init_fp8_kv_cache(
                bsz, self.num_key_value_heads, kv_seq_len, self.head_dim,
                device=query_states.device
            )
            key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                           key_states, value_states)
            past_key_value = (key_states, value_states)
    else:
        k_cache, v_cache = past_key_value
        key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                       key_states, value_states)
        kv_seq_len = key_states.shape[-2]
        past_key_value = (key_states, value_states)

        if not use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                                   attention_mask, cache_position,
                                                   bsz, q_len, kv_seq_len,
                                                   self.head_dim, self.num_heads, output_attentions)
        else:
            import xe_addons
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states, attention_mask)
            attn_weights = None

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def llama_attention_forward_4_31_original(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(getattr(self, "q_proj", None),
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp

    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        kv_seq_len = past_key_value[0].shape[-2]
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
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

    else:
        if self.config.pretraining_tp > 1:
            key_value_slicing = ((self.num_key_value_heads * self.head_dim) //
                                 self.config.pretraining_tp)
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim)
                                                    // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i])
                            for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i])
                          for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i])
                            for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            if fp16_fusion_check(getattr(self, "q_proj", None), hidden_states, self.training) and \
                    hidden_size == 4096 and self.q_proj.out_features == self.k_proj.out_features:
                # only use mm_qkv_out on pvc for llama-7b
                if not hasattr(self, "qkv_proj_weight"):
                    self.qkv_proj_weight = torch.stack([self.q_proj.weight,
                                                        self.k_proj.weight,
                                                        self.v_proj.weight]).contiguous()
                    self.q_proj.weight.data = self.qkv_proj_weight[0, :, :]
                    self.k_proj.weight.data = self.qkv_proj_weight[1, :, :]
                    self.v_proj.weight.data = self.qkv_proj_weight[2, :, :]
                    torch.xpu.empty_cache()
                query_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                key_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                         dtype=hidden_states.dtype, device=hidden_states.device)
                value_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                torch.ops.torch_ipex.mm_qkv_out(
                    hidden_states, self.qkv_proj_weight, None,
                    query_states, key_states, value_states
                )
            else:
                if should_use_xetla_mm_qkv(self, device):
                    if not hasattr(self, "qkv_proj_qweight"):
                        self.qkv_proj_qweight = fuse_qkv_weight_xetla(self.q_proj,
                                                                      self.k_proj,
                                                                      self.v_proj,
                                                                      self.q_proj.weight.qtype,)
                    import xe_linear
                    q_out_len = self.q_proj.out_len
                    k_out_len = self.k_proj.out_len
                    v_out_len = self.v_proj.out_len
                    qkv_states = xe_linear.mm_xetla(hidden_states, self.qkv_proj_qweight,
                                                    self.q_proj.weight.qtype)
                    query_states = qkv_states[:, :, :q_out_len]
                    key_states = qkv_states[:, :, q_out_len:q_out_len + k_out_len]
                    value_states = qkv_states[:, :, q_out_len + k_out_len:]
                else:
                    if hasattr(self, "q_proj"):
                        query_states = self.q_proj(hidden_states)
                        key_states = self.k_proj(hidden_states)
                        value_states = self.v_proj(hidden_states)
                    else:
                        qkv = self.qkv_proj(hidden_states)
                        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads,
                                       self.head_dim)
                        query_states, key_states, value_states = \
                            qkv.split([self.num_heads,
                                       self.num_key_value_heads,
                                       self.num_key_value_heads],
                                      dim=2)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if use_fuse_rope:
            import xe_addons
            xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                           query_states, key_states)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "llama")

        if past_key_value is not None:
            # reuse k, v, self_attention
            cache_k = past_key_value[0]
            cache_v = past_key_value[1]
            if not enough_kv_room:
                # allocate new
                new_cache_k, new_cache_v = extend_kv_cache(
                    bsz,
                    self.num_key_value_heads,  # Support GQA
                    self.head_dim,
                    cache_k.size(2),
                    kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                    dtype=cache_k.dtype,
                    device=device
                )
                new_cache_k[:] = cache_k
                new_cache_v[:] = cache_v
                cache_k = new_cache_k
                cache_v = new_cache_v

            key_states, value_states = append_kv_cache(cache_k, cache_v, key_states, value_states)

        elif use_cache:
            max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_key_states, new_value_states = init_kv_cache(bsz,
                                                             self.num_key_value_heads,
                                                             self.head_dim,
                                                             kv_seq_len,
                                                             max_cache_length,
                                                             dtype=key_states.dtype,
                                                             device=device)
            new_key_states[:] = key_states
            new_value_states[:] = value_states
            key_states = new_key_states
            value_states = new_value_states

    past_key_value = (key_states, value_states) if use_cache else None

    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, attention_mask):
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(query_states.to(device, dtype=torch.float16),
                                                     key_states.to(device, dtype=torch.float16),
                                                     value_states.to(device, dtype=torch.float16),
                                                     is_causal=True)
        attn_weights = None
    elif not self.training and not hidden_states.requires_grad and \
            use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import xe_addons
        attn_output = xe_addons.sdp(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # otherwise, use native attention
        attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                               attention_mask, cache_position,
                                               bsz, q_len, kv_seq_len,
                                               self.head_dim, self.num_heads, output_attentions)
    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def llama_attention_selective_batching_forward_4_31(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Minimize this value to reduce memory allocation.
    VLLM_KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get('VLLM_KV_CACHE_ALLOC_BLOCK', 64))
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype
    # TODO: consider this later - flash attention
    # if not self.training and not hidden_states.requires_grad:
    #     fsdp_flag = use_flash_attention(hidden_states)
    # else:
    #     fsdp_flag = False
    # if fsdp_flag and q_len > 1:
    #     attention_dtype = torch.float16  # use fp16 for flash attention
    # else:
    #     attention_dtype = original_dtype

    attention_dtype = original_dtype

    # TODO: decoding fast path
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = past_key_value is not None and is_enough_kv_cache_room_4_31(past_key_value[0])
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(getattr(self, "q_proj", None),
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp

    updated_past_key_values = []
    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
    if decoding_fast_path:
        past_k = past_key_value[0][0]
        past_v = past_key_value[0][1]
        kv_seq_len = past_k.shape[-2]
        if not enough_kv_room:
            new_cache_k, new_cache_v = extend_kv_cache(1,
                                                       self.num_key_value_heads,  # Support GQA
                                                       self.head_dim,
                                                       kv_seq_len,
                                                       kv_seq_len +
                                                       VLLM_KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=past_k.dtype,
                                                       device=device)
            new_cache_k[:] = past_k
            new_cache_v[:] = past_v
            past_k = new_cache_k
            past_v = new_cache_v
        hidden_states = hidden_states.view(1, -1)
        import xe_linear
        query_states, key_states, value_states = xe_linear.forward_qkv(hidden_states,
                                                                       self.q_proj.weight,
                                                                       self.k_proj.weight,
                                                                       self.v_proj.weight,
                                                                       position_ids,
                                                                       past_k, past_v,
                                                                       self.q_proj.weight.qtype,
                                                                       self.v_proj.weight.qtype,
                                                                       kv_seq_len,
                                                                       self.head_dim,
                                                                       self.rotary_emb.base,
                                                                       )
        kv_seq_len += 1
    else:
        if self.config.pretraining_tp > 1:
            invalidInputError(False, f"vLLM: config.pretraining_tp > 1 not supported yet")
        else:
            if hasattr(self, "q_proj"):
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
            else:
                qkv = self.qkv_proj(hidden_states)
                qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads,
                               self.head_dim)
                query_states, key_states, value_states = qkv.split([self.num_heads,
                                                                    self.num_key_value_heads,
                                                                    self.num_key_value_heads],
                                                                   dim=2)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += max(kv_pair[0].shape[-2] for kv_pair in past_key_value)

        if use_fuse_rope:
            import xe_addons
            xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                           query_states, key_states)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "llama")

        if past_key_value is not None:
            batched_attention_output = []
            # print(f"type of attention_mask is {type(attention_mask)}")
            for batch in range(bsz):
                enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value[batch])
                past_k, past_v = past_key_value[batch]
                current_kv_len = past_k.shape[-2] + 1
                if not enough_kv_room:
                    # allocate new
                    new_cache_k, new_cache_v = extend_kv_cache(1,
                                                               self.num_key_value_heads,
                                                               self.head_dim,
                                                               past_k.size(2),
                                                               current_kv_len +
                                                               VLLM_KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                               dtype=past_k.dtype,
                                                               device=device)
                    new_cache_k[:] = past_k
                    new_cache_v[:] = past_v
                    past_k = new_cache_k
                    past_v = new_cache_v

                current_key_states = key_states[batch: batch + 1, :, :, :]
                current_value_states = value_states[batch: batch + 1, :, :, :]
                current_key_states, current_value_states = append_kv_cache(past_k,
                                                                           past_v,
                                                                           current_key_states,
                                                                           current_value_states)
                updated_past_key_values.append((current_key_states, current_value_states))

                current_key_states = repeat_kv(current_key_states, self.num_key_value_groups)
                current_value_states = repeat_kv(current_value_states, self.num_key_value_groups)

                cache_position = None
                current_query_states = query_states[batch: batch + 1, :, :, :]
                attn_output, attn_weights = native_sdp(current_query_states,
                                                       current_key_states,
                                                       current_value_states,
                                                       attention_mask[batch],
                                                       cache_position,
                                                       1,
                                                       1,
                                                       current_kv_len,
                                                       self.head_dim,
                                                       self.num_heads,
                                                       output_attentions)
                if attn_output.size() != (1, self.num_heads, 1, self.head_dim):
                    invalidInputError(False,
                                      f"`attn_output` should be of size "
                                      f"{(1, self.num_heads, 1, self.head_dim)}, but is"
                                      f" {attn_output.size()}")
                batched_attention_output.append(attn_output)
            # For loop ends
            # TODO: handle attention_weights later
            attn_output = torch.concat(batched_attention_output, dim=0)
            batched_attention_output.clear()
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                invalidInputError(False,
                                  f"`attn_output` should be of size "
                                  f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                                  f" {attn_output.size()}")
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, updated_past_key_values

    # Assume always use_cache
    # prefill or decoding fast path
    for batch in range(bsz):
        updated_past_key_values.append((key_states[batch: batch + 1, :, :, :],
                                        value_states[batch: batch+1, :, :, :]))

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups).to(device,
                                                                     dtype=attention_dtype)
    value_states = repeat_kv(value_states, self.num_key_value_groups).to(device,
                                                                         dtype=attention_dtype)
    # Can also happens for decoding fast path
    if isinstance(attention_mask, list):
        # For decoding fast path
        attention_mask = attention_mask[0]
    cache_position = None
    attn_output, attn_weights = native_sdp(query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           cache_position,
                                           bsz,
                                           q_len,
                                           kv_seq_len,
                                           self.head_dim,
                                           self.num_heads,
                                           output_attentions)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(False,
                          f"`attn_output` should be of size "
                          f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                          f" {attn_output.size()}")
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
    return attn_output.to(original_dtype), attn_weights, updated_past_key_values


def llama_attention_forward_4_41(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    if use_quantize_kv_cache(get_q_proj_or_qkv_proj(self), hidden_states,
                             self.num_key_value_groups):
        forward_function = llama_attention_forward_4_41_quantized
    else:
        forward_function = llama_attention_forward_4_41_original
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        kwargs=kwargs
    )


def llama_attention_forward_4_41_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    from ipex_llm.transformers.kv import DynamicCompressCache
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(getattr(self, "q_proj", None),
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp
    # [CompressKV]
    use_compresskv = isinstance(past_key_value, DynamicCompressCache)

    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        tmp_cache_k, tmp_cache_v = init_kv_cache(
            bsz,
            self.num_key_value_heads,
            self.head_dim,
            0,
            1,
            dtype=hidden_states.dtype,
            device=device
        )
        import xe_linear
        query_states, key_states, value_states = xe_linear.forward_qkv(hidden_states,
                                                                       self.q_proj.weight,
                                                                       self.k_proj.weight,
                                                                       self.v_proj.weight,
                                                                       position_ids,
                                                                       tmp_cache_k, tmp_cache_v,
                                                                       self.q_proj.weight.qtype,
                                                                       self.v_proj.weight.qtype,
                                                                       0,
                                                                       self.head_dim,
                                                                       self.rotary_emb.base,)
    else:
        if hasattr(self, "q_proj"):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            query_states, key_states, value_states = qkv.split([self.num_heads,
                                                                self.num_key_value_heads,
                                                                self.num_key_value_heads], dim=2)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                invalidInputError(
                    False,
                    f"The cache structure has changed since version v4.36."
                    f" If you are using {self.__class__.__name__} "
                    f"for auto-regressive decoding with k/v caching,"
                    f" please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if use_fuse_rope:
            import xe_addons
            xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                           query_states, key_states)
        else:
            if cache_position is not None:
                # for transformers 4.38.0
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama2")
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama")
    kv_seq_len = key_states.shape[-2]

    if len(past_key_value.key_cache) <= self.layer_idx:
        repeated_key_states = repeat_kv(key_states, self.num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, self.num_key_value_groups)
        if use_cache:
            cache_kwargs = None
            # [CompressKV]
            if use_compresskv:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    query_states, attention_mask, self.num_key_value_groups,
                    self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states,
                                                                 self.layer_idx, cache_kwargs)
        if use_cache and use_sdp_causal(q_len, kv_seq_len, self.head_dim,
                                        query_states, self.training):
            import xe_addons
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        elif should_split_qkv_tensor(query_states, bsz, self.num_heads,
                                     q_len, kv_seq_len, output_attentions):
            attn_output, _ = native_sdp_split_qkv_tensor(query_states, repeated_key_states,
                                                         repeated_value_states,
                                                         attention_mask, cache_position,
                                                         bsz, q_len, kv_seq_len, self.head_dim,
                                                         self.num_heads)
        else:
            attn_weights = torch.matmul(query_states, repeated_key_states
                                        .transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size "
                    f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if cache_position is not None:
                    # for transformers 4.38.0
                    causal_mask = attention_mask[:, :, :, : kv_seq_len]
                    attn_weights = attn_weights + causal_mask
                else:
                    attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                    if attention_mask.size() != attn_mask_size:
                        invalidInputError(False,
                                          f"Attention mask should be of size {attn_mask_size}, "
                                          f"but is {attention_mask.size()}")
                    attn_weights = attn_weights + attention_mask

            if kv_seq_len >= 2048 or bsz >= 64:
                # for memory considerations, do not upcast attention to fp32
                # for long sequences or large batches
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                     dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, repeated_value_states)
    else:
        cache_kwargs = None  # Specific to RoPE models
        # [CompressKV]
        if use_compresskv:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx,
                query_states, attention_mask, self.num_key_value_groups,
                self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states,
                                                             self.layer_idx, cache_kwargs)
        kv_seq_len = key_states.shape[-2]
        if not use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            key_states = repeat_kv(key_states, self.num_key_value_groups)\
                .to(device, dtype=query_states.dtype)
            value_states = repeat_kv(value_states, self.num_key_value_groups)\
                .to(device, dtype=query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size"
                    f" {(bsz, self.num_heads, q_len, kv_seq_len)},"
                    f" but is {attn_weights.size()}"
                )

            if attention_mask is not None:
                if cache_position is not None:
                    # for transformers 4.38.0
                    causal_mask = attention_mask[:, :, :, : kv_seq_len]
                    attn_weights = attn_weights + causal_mask
                else:
                    attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                    if attention_mask.size() != attn_mask_size:
                        invalidInputError(False,
                                          f"Attention mask should be of size {attn_mask_size}, "
                                          f"but is {attention_mask.size()}")
                    attn_weights = attn_weights + attention_mask

            if kv_seq_len >= 2048 or bsz >= 64:
                # for memory considerations, do not upcast attention to fp32
                # for long sequences or large batches
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                     dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import xe_addons
            if cache_position is not None:
                new_attn_mask = attention_mask[:, :, :, 0:kv_seq_len]
            else:
                new_attn_mask = attention_mask

            # [CompressKV]
            if use_compresskv:
                new_attn_mask = get_compresskv_attn_mask(key_states,
                                                         new_attn_mask)
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states, new_attn_mask)
            attn_weights = None

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(
            False,
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)},"
            f" but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size
                                                 // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i],
                                    o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attention_forward_4_41_original(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    from ipex_llm.transformers.kv import DynamicCompressCache
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    # [CompressKV]
    use_compresskv = isinstance(past_key_value, DynamicCompressCache)

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(getattr(self, "q_proj", None),
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp

    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
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
        # [CompressKV]
        if use_compresskv:
            past_key_value.update_seen_tokens(self.layer_idx, q_len)
            kv_seq_len = past_key_value.get_seq_length()
        elif self.layer_idx == 0:
            past_key_value._seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states

    else:
        if self.config.pretraining_tp > 1:
            key_value_slicing = ((self.num_key_value_heads * self.head_dim) //
                                 self.config.pretraining_tp)
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim)
                                                    // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i])
                            for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i])
                          for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i])
                            for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            if fp16_fusion_check(getattr(self, "q_proj", None), hidden_states, self.training) and \
                    hidden_size == 4096 and self.q_proj.out_features == self.k_proj.out_features:
                # only use mm_qkv_out on pvc for llama-7b
                if not hasattr(self, "qkv_proj_weight"):
                    self.qkv_proj_weight = torch.stack([self.q_proj.weight,
                                                        self.k_proj.weight,
                                                        self.v_proj.weight]).contiguous()
                    self.q_proj.weight.data = self.qkv_proj_weight[0, :, :]
                    self.k_proj.weight.data = self.qkv_proj_weight[1, :, :]
                    self.v_proj.weight.data = self.qkv_proj_weight[2, :, :]
                    torch.xpu.empty_cache()
                query_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                key_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                         dtype=hidden_states.dtype, device=hidden_states.device)
                value_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                torch.ops.torch_ipex.mm_qkv_out(
                    hidden_states, self.qkv_proj_weight, None,
                    query_states, key_states, value_states
                )
            else:
                if should_use_xetla_mm_qkv(self, device):
                    if not hasattr(self, "qkv_proj_qweight"):
                        self.qkv_proj_qweight = fuse_qkv_weight_xetla(self.q_proj,
                                                                      self.k_proj,
                                                                      self.v_proj,
                                                                      self.q_proj.weight.qtype,)
                    import xe_linear
                    q_out_len = self.q_proj.out_len
                    k_out_len = self.k_proj.out_len
                    v_out_len = self.v_proj.out_len
                    qkv_states = xe_linear.mm_xetla(hidden_states,
                                                    self.qkv_proj_qweight,
                                                    self.q_proj.weight.qtype)
                    query_states = qkv_states[:, :, :q_out_len]
                    key_states = qkv_states[:, :, q_out_len:q_out_len + k_out_len]
                    value_states = qkv_states[:, :, q_out_len + k_out_len:]
                else:
                    if hasattr(self, "q_proj"):
                        query_states = self.q_proj(hidden_states)
                        key_states = self.k_proj(hidden_states)
                        value_states = self.v_proj(hidden_states)
                    else:
                        qkv = self.qkv_proj(hidden_states)
                        qkv = qkv.view(bsz, q_len,
                                       self.num_heads + 2 * self.num_key_value_heads,
                                       self.head_dim)
                        query_states, key_states, value_states = \
                            qkv.split([self.num_heads,
                                       self.num_key_value_heads,
                                       self.num_key_value_heads],
                                      dim=2)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
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
            import xe_addons
            xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                           query_states, key_states)
        else:
            if cache_position is not None:
                # for transformers 4.38.0
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama2")
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama")

        if past_key_value is not None:
            if use_compresskv:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    query_states, attention_mask, self.num_key_value_groups,
                    self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH)
            else:
                # update the number of seen tokens
                if self.layer_idx == 0:
                    past_key_value._seen_tokens += key_states.shape[-2]

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

    if attention_mask is not None:
        new_attention_mask = attention_mask[:, :, :, 0:kv_seq_len]
    else:
        new_attention_mask = attention_mask

    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, new_attention_mask):
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # now only use flash attention for first token
        attn_output = F.scaled_dot_product_attention(query_states.to(device, dtype=torch.float16),
                                                     key_states.to(device, dtype=torch.float16),
                                                     value_states.to(device, dtype=torch.float16),
                                                     is_causal=True)
        attn_weights = None
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim,
                        query_states, self.training):
        import xe_addons
        attn_output = xe_addons.sdp_causal(query_states, key_states.contiguous(),
                                           value_states.contiguous(), new_attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    elif not self.training and not hidden_states.requires_grad and \
            use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import xe_addons
        # [CompressKV]
        if use_compresskv:
            new_attention_mask = get_compresskv_attn_mask(key_states,
                                                          new_attention_mask)
        attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                    new_attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # otherwise, use native attention
        if query_states.device.type == "xpu":
            attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                                   new_attention_mask, cache_position,
                                                   bsz, q_len, kv_seq_len,
                                                   self.head_dim, self.num_heads, output_attentions)
        else:
            # CPU path
            if not output_attentions:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=new_attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    # The q_len > 1 is necessary to match with
                    # AttentionMaskConverter.to_causal_4d that
                    # does not create a causal mask in case q_len == 1.
                    is_causal=self.is_causal and new_attention_mask is None and q_len > 1,
                )
            else:
                attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                                       new_attention_mask, cache_position,
                                                       bsz, q_len, kv_seq_len,
                                                       self.head_dim,
                                                       self.num_heads, output_attentions)

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def llama_attention_forward_4_38(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    if use_quantize_kv_cache(get_q_proj_or_qkv_proj(self), hidden_states,
                             self.num_key_value_groups):
        forward_function = llama_attention_forward_4_38_quantized
    else:
        forward_function = llama_attention_forward_4_38_original
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        kwargs=kwargs
    )


def llama_attention_forward_4_38_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    from ipex_llm.transformers.kv import DynamicCompressCache
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(getattr(self, "q_proj", None),
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp

    # [CompressKV]
    use_compresskv = isinstance(past_key_value, DynamicCompressCache)

    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        tmp_cache_k, tmp_cache_v = init_kv_cache(
            bsz,
            self.num_key_value_heads,
            self.head_dim,
            0,
            1,
            dtype=hidden_states.dtype,
            device=device
        )
        import xe_linear
        query_states, key_states, value_states = xe_linear.forward_qkv(hidden_states,
                                                                       self.q_proj.weight,
                                                                       self.k_proj.weight,
                                                                       self.v_proj.weight,
                                                                       position_ids,
                                                                       tmp_cache_k, tmp_cache_v,
                                                                       self.q_proj.weight.qtype,
                                                                       self.v_proj.weight.qtype,
                                                                       0,
                                                                       self.head_dim,
                                                                       self.rotary_emb.base,)
    else:
        if hasattr(self, "q_proj"):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            query_states, key_states, value_states = qkv.split([self.num_heads,
                                                                self.num_key_value_heads,
                                                                self.num_key_value_heads], dim=2)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                invalidInputError(
                    False,
                    f"The cache structure has changed since version v4.36."
                    f" If you are using {self.__class__.__name__} "
                    f"for auto-regressive decoding with k/v caching,"
                    f" please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if use_fuse_rope:
            import xe_addons
            xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                           query_states, key_states)
        else:
            if cache_position is not None:
                # for transformers 4.38.0
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama2")
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama")
    kv_seq_len = key_states.shape[-2]

    if len(past_key_value.key_cache) <= self.layer_idx:
        repeated_key_states = repeat_kv(key_states, self.num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, self.num_key_value_groups)
        if use_cache:
            cache_kwargs = None
            # [CompressKV]
            if use_compresskv:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    query_states, attention_mask, self.num_key_value_groups,
                    self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states,
                                                                 self.layer_idx, cache_kwargs)

        if use_cache and use_sdp_causal(q_len, kv_seq_len, self.head_dim,
                                        query_states, self.training):
            import xe_addons
            attn_output = xe_addons.sdp_fp8_causal(query_states, key_states,
                                                   value_states, attention_mask)
        elif should_split_qkv_tensor(query_states, bsz, self.num_heads,
                                     q_len, kv_seq_len, output_attentions):
            attn_output, _ = native_sdp_split_qkv_tensor(query_states, repeated_key_states,
                                                         repeated_value_states,
                                                         attention_mask, cache_position,
                                                         bsz, q_len, kv_seq_len, self.head_dim,
                                                         self.num_heads)
        else:
            attn_weights = torch.matmul(query_states, repeated_key_states
                                        .transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size "
                    f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if cache_position is not None:
                    # for transformers 4.38.0
                    causal_mask = attention_mask[:, :, cache_position, : kv_seq_len]
                    attn_weights = attn_weights + causal_mask
                else:
                    attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                    if attention_mask.size() != attn_mask_size:
                        invalidInputError(False,
                                          f"Attention mask should be of size {attn_mask_size}, "
                                          f"but is {attention_mask.size()}")
                    attn_weights = attn_weights + attention_mask

            if kv_seq_len >= 2048 or bsz >= 64:
                # for memory considerations, do not upcast attention to fp32
                # for long sequences or large batches
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                     dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, repeated_value_states)
    else:
        cache_kwargs = None  # Specific to RoPE models
        # [CompressKV]
        if use_compresskv:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx,
                query_states, attention_mask, self.num_key_value_groups,
                self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH)
        else:
            key_states, value_states = past_key_value.update(key_states, value_states,
                                                             self.layer_idx, cache_kwargs)
        kv_seq_len = key_states.shape[-2]
        if not use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            key_states = repeat_kv(key_states, self.num_key_value_groups)\
                .to(device, dtype=query_states.dtype)
            value_states = repeat_kv(value_states, self.num_key_value_groups)\
                .to(device, dtype=query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size"
                    f" {(bsz, self.num_heads, q_len, kv_seq_len)},"
                    f" but is {attn_weights.size()}"
                )

            if attention_mask is not None:
                if cache_position is not None:
                    # for transformers 4.38.0
                    causal_mask = attention_mask[:, :, cache_position, : kv_seq_len]
                    attn_weights = attn_weights + causal_mask
                else:
                    attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                    if attention_mask.size() != attn_mask_size:
                        invalidInputError(False,
                                          f"Attention mask should be of size {attn_mask_size}, "
                                          f"but is {attention_mask.size()}")
                    attn_weights = attn_weights + attention_mask

            if kv_seq_len >= 2048 or bsz >= 64:
                # for memory considerations, do not upcast attention to fp32
                # for long sequences or large batches
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                     dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import xe_addons
            if cache_position is not None:
                new_attn_mask = attention_mask[:, :, kv_seq_len-q_len:kv_seq_len, 0:kv_seq_len]
            else:
                new_attn_mask = attention_mask

            # [CompressKV]
            if use_compresskv:
                new_attn_mask = get_compresskv_attn_mask(key_states,
                                                         new_attn_mask)
            attn_output = xe_addons.sdp_fp8(query_states, key_states, value_states, new_attn_mask)
            attn_weights = None

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(
            False,
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)},"
            f" but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size
                                                 // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i],
                                    o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attention_forward_4_38_original(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[List[torch.FloatTensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.FloatTensor]]]:
    from ipex_llm.transformers.kv import DynamicCompressCache
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    # [CompressKV]
    use_compresskv = isinstance(past_key_value, DynamicCompressCache)

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, seq_len=q_len)
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = use_decoding_fast_path(getattr(self, "q_proj", None),
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len,
                                                llama_decoding_fast_path_qtype_check) and no_tp

    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
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
        # [CompressKV]
        if use_compresskv:
            past_key_value.update_seen_tokens(self.layer_idx, q_len)
            kv_seq_len = past_key_value.get_seq_length()
        elif self.layer_idx == 0:
            past_key_value.seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states
    else:
        if self.config.pretraining_tp > 1:
            key_value_slicing = ((self.num_key_value_heads * self.head_dim) //
                                 self.config.pretraining_tp)
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim)
                                                    // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i])
                            for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i])
                          for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i])
                            for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            if fp16_fusion_check(getattr(self, "q_proj", None), hidden_states, self.training) and \
                    hidden_size == 4096 and self.q_proj.out_features == self.k_proj.out_features:
                # only use mm_qkv_out on pvc for llama-7b
                if not hasattr(self, "qkv_proj_weight"):
                    self.qkv_proj_weight = torch.stack([self.q_proj.weight,
                                                        self.k_proj.weight,
                                                        self.v_proj.weight]).contiguous()
                    self.q_proj.weight.data = self.qkv_proj_weight[0, :, :]
                    self.k_proj.weight.data = self.qkv_proj_weight[1, :, :]
                    self.v_proj.weight.data = self.qkv_proj_weight[2, :, :]
                    torch.xpu.empty_cache()
                query_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                key_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                         dtype=hidden_states.dtype, device=hidden_states.device)
                value_states = torch.empty(bsz, q_len, self.qkv_proj_weight.shape[-1],
                                           dtype=hidden_states.dtype, device=hidden_states.device)
                torch.ops.torch_ipex.mm_qkv_out(
                    hidden_states, self.qkv_proj_weight, None,
                    query_states, key_states, value_states
                )
            else:
                if should_use_xetla_mm_qkv(self, device):
                    if not hasattr(self, "qkv_proj_qweight"):
                        self.qkv_proj_qweight = fuse_qkv_weight_xetla(self.q_proj,
                                                                      self.k_proj,
                                                                      self.v_proj,
                                                                      self.q_proj.weight.qtype,)
                    import xe_linear
                    q_out_len = self.q_proj.out_len
                    k_out_len = self.k_proj.out_len
                    v_out_len = self.v_proj.out_len
                    qkv_states = xe_linear.mm_xetla(hidden_states,
                                                    self.qkv_proj_qweight,
                                                    self.q_proj.weight.qtype)
                    query_states = qkv_states[:, :, :q_out_len]
                    key_states = qkv_states[:, :, q_out_len:q_out_len + k_out_len]
                    value_states = qkv_states[:, :, q_out_len + k_out_len:]
                else:
                    if hasattr(self, "q_proj"):
                        query_states = self.q_proj(hidden_states)
                        key_states = self.k_proj(hidden_states)
                        value_states = self.v_proj(hidden_states)
                    else:
                        qkv = self.qkv_proj(hidden_states)
                        qkv = qkv.view(bsz, q_len,
                                       self.num_heads + 2 * self.num_key_value_heads,
                                       self.head_dim)
                        query_states, key_states, value_states = \
                            qkv.split([self.num_heads,
                                       self.num_key_value_heads,
                                       self.num_key_value_heads],
                                      dim=2)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
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
            import xe_addons
            xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                           query_states, key_states)
        else:
            if cache_position is not None:
                # for transformers 4.38.0
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama2")
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                                cos, sin, position_ids, "llama")

        if past_key_value is not None:
            if use_compresskv:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    query_states, attention_mask, self.num_key_value_groups,
                    self.config, enough_kv_room, KV_CACHE_ALLOC_BLOCK_LENGTH)
            else:
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

    if cache_position is not None:
        new_attention_mask = attention_mask[:, :, kv_seq_len - q_len:kv_seq_len, 0:kv_seq_len]
    else:
        new_attention_mask = attention_mask

    if not self.training and not hidden_states.requires_grad and \
            use_flash_attention(query_states, key_states, new_attention_mask):
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # now only use flash attention for first token
        attn_output = F.scaled_dot_product_attention(query_states.to(device, dtype=torch.float16),
                                                     key_states.to(device, dtype=torch.float16),
                                                     value_states.to(device, dtype=torch.float16),
                                                     is_causal=True)
        attn_weights = None
    elif use_sdp_causal(q_len, kv_seq_len, self.head_dim,
                        query_states, self.training):
        import xe_addons
        attn_output = xe_addons.sdp_causal(query_states, key_states.contiguous(),
                                           value_states.contiguous(), new_attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    elif not self.training and not hidden_states.requires_grad and \
            use_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import xe_addons
        # [CompressKV]
        if use_compresskv:
            new_attention_mask = get_compresskv_attn_mask(key_states,
                                                          new_attention_mask)
        attn_output = xe_addons.sdp(query_states, key_states, value_states,
                                    new_attention_mask)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # otherwise, use native attention
        if query_states.device.type == "xpu":
            attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                                   new_attention_mask, cache_position,
                                                   bsz, q_len, kv_seq_len,
                                                   self.head_dim, self.num_heads, output_attentions)
        else:
            # CPU path
            if not output_attentions:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=new_attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    # The q_len > 1 is necessary to match with
                    # AttentionMaskConverter.to_causal_4d that
                    # does not create a causal mask in case q_len == 1.
                    is_causal=self.is_causal and new_attention_mask is None and q_len > 1,
                )
            else:
                attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                                       new_attention_mask, cache_position,
                                                       bsz, q_len, kv_seq_len,
                                                       self.head_dim,
                                                       self.num_heads, output_attentions)

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def native_sdp(query, key, value, attention_mask, cache_position,
               bsz, q_len, kv_seq_len, head_dim, num_heads, output_attentions):
    if should_split_qkv_tensor(query, bsz, num_heads, q_len, kv_seq_len, output_attentions):
        return native_sdp_split_qkv_tensor(query, key, value, attention_mask, cache_position,
                                           bsz, q_len, kv_seq_len, head_dim, num_heads)
    else:
        attn_weights = torch.matmul(query.to(key.dtype),
                                    key.transpose(2, 3)) / math.sqrt(head_dim)

        attn_weights_size = (bsz, num_heads, q_len, kv_seq_len)
        if attn_weights.size() != attn_weights_size:
            invalidInputError(False,
                              f"Attention weights should be of size {attn_weights_size}, "
                              f"but is {attn_weights.size()}")

        if attention_mask is not None:
            if cache_position is not None:
                # for transformers 4.38.0
                causal_mask = attention_mask[:, :, cache_position, : kv_seq_len]
                attn_weights = attn_weights + causal_mask
            else:
                attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                if attention_mask.size() != attn_mask_size:
                    invalidInputError(False,
                                      f"Attention mask should be of size {attn_mask_size}, "
                                      f"but is {attention_mask.size()}")
                attn_weights = attn_weights + attention_mask

        if kv_seq_len >= 2048 or bsz >= 64:
            # for memory considerations, do not upcast attention to fp32
            # for long sequences or large batches
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        else:
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                 dtype=torch.float32).to(value.dtype)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def native_sdp_split_qkv_tensor(query, key, value, attention_mask, cache_position,
                                bsz, q_len, kv_seq_len, head_dim, num_heads):
    block_size = 8
    query_split = torch.split(query.to(key.dtype), block_size, dim=1)
    key_split = torch.split(key.transpose(2, 3), block_size, dim=1)
    value_split = torch.split(value, block_size, dim=1)
    attn_outputs = []
    for q, k, v in zip(query_split, key_split, value_split):
        attn_weights_split = torch.matmul(q, k) / math.sqrt(head_dim)
        block_actual_size = attn_weights_split.size(1)
        attn_weights_split_size = (bsz, block_actual_size, q_len, kv_seq_len)
        if attn_weights_split.size() != attn_weights_split_size:
            invalidInputError(False,
                              f"Splitted attention weights should be of size "
                              f"{attn_weights_split_size}, but is {attn_weights_split.size()}")

        if attention_mask is not None:
            if cache_position is not None:
                # for transformers 4.38.0
                causal_mask = attention_mask[:, :, cache_position, : kv_seq_len]
                attn_weights_split = attn_weights_split + causal_mask
            else:
                attn_mask_size = (bsz, 1, q_len, kv_seq_len)
                if attention_mask.size() != attn_mask_size:
                    invalidInputError(False,
                                      f"Attention mask should be of size {attn_mask_size}, "
                                      f"but is {attention_mask.size()}")
                attn_weights_split = attn_weights_split + attention_mask
        attn_weights_split = nn.functional.softmax(attn_weights_split, dim=-1)
        attn_outputs.append(torch.matmul(attn_weights_split, v))
    attn_output = torch.cat(attn_outputs, dim=1)
    return attn_output.to(key.dtype), None


def llama_model_selective_batching_forward_4_31(
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
    if output_attentions is not None:
        output_attentions = output_attentions
    else:
        output_attentions = self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False,
                          "You cannot specify both decoder_input_ids"
                          " and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        invalidInputError(False,
                          "You have to specify either "
                          "decoder_input_ids or decoder_inputs_embeds")

    # seq_length_with_past = seq_length
    past_key_values_length = 0

    # The original position_ids in the format of [1, 1]
    # However, this only applies when kv_len is the same for all the sequences
    # We should set it to format of [batch, position_id]
    # TODO: validate correctness
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        invalidInputError(False,
                          "vLLM: position_ids should never be None")
    else:
        # print(f"Original position_ids is {position_ids}")
        position_ids = position_ids.view(-1, seq_length)
        # print(f"after position_ids is {position_ids}")
    # if past_key_values is None:
    #     # For prefill
    #     position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
    #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    # else:
    #     past_key_values_length = []
    #     for sequence_kv in past_key_values[0]:
    #         key = sequence_kv[0]
    #         past_key_values_length.append(key.shape[-2])
    #     position_ids = torch.tensor(past_key_values_length, dtype=torch.long, device=device)
    #     position_ids = position_ids.unsqueeze(0).view(-1, 1)

    if past_key_values is not None:
        # past_key_values in the format of num_layers x num_seqs x 2
        # TODO: this may be incorrect
        past_key_values_length = past_key_values[0][0][0].shape[2]
        # seq_length_with_past = seq_length_with_past + past_key_values_length

    # if position_ids is None:
    #     device = input_ids.device if input_ids is not None else inputs_embeds.device
    #     # [start, end)
    #     position_ids = torch.arange(
    #         past_key_values_length, seq_length +
    #         past_key_values_length, dtype=torch.long, device=device
    #     )
    #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    # else:
    #     position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        invalidInputError(False, "attention_mask should never be None")
    # print(f"attention_mask before expanding: {attention_mask}")
    if past_key_values is None:
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
    else:
        i = 0
        for attn_mask in attention_mask:
            past_key_value_length = past_key_values[0][i][0].shape[2]
            new_mask = self._prepare_decoder_attention_mask(
                attn_mask, (1, seq_length), inputs_embeds, past_key_value_length
            )
            attention_mask[i] = new_mask
            i += 1

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        invalidInputError(False, "gradient_checkpointing is not supported")

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                     if v is not None)  # noqa
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# For training
def llama_attention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    use_fast_rope = should_use_fast_rope(self, hidden_states, position_ids)

    # Check for inference
    if use_cache and past_key_value is not None and q_len == 1:
        A, past_key_value = llama_attention_forward_4_31(
            self,
            hidden_states,
            past_key_value,
            position_ids,
        )
        return A, None, past_key_value

    if self.config.pretraining_tp > 1:
        key_value_slicing = ((self.num_key_value_heads * self.head_dim) //
                             self.config.pretraining_tp)
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i])
                        for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i])
                      for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i])
                        for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        if hasattr(self, "q_proj"):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            query_states, key_states, value_states = qkv.split([self.num_heads,
                                                                self.num_key_value_heads,
                                                                self.num_key_value_heads], dim=2)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if use_fast_rope:
        from ipex_llm.transformers.layers.rope_embedding import apply_fast_rope_embedding
        query_states, key_states = apply_fast_rope_embedding(query_states,
                                                             key_states,
                                                             position_ids,
                                                             "llama")
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids, "llama")

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    cache_position = None
    attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                           attention_mask, cache_position,
                                           bsz, q_len, kv_seq_len,
                                           self.head_dim, self.num_heads, output_attentions)

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_model_forward_4_41_internal(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]]=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions\
        if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
# retrieve input_ids and inputs_embeds

    if (input_ids is None) ^ (inputs_embeds is not None):
        invalidInputError(False,
                          f"You cannot specify both input_ids and inputs_embeds at the same time,"
                          f" and must specify either one")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing.",
            "Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    # kept for BC (non `Cache` `past_key_values` inputs)
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() \
            if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens
            + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
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

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            # bigdl-llm changes:
            curr_device = decoder_layer.input_layernorm.weight.device
            if causal_mask is not None:
                causal_mask = causal_mask.to(curr_device)
            if position_ids is not None:
                position_ids = position_ids.to(curr_device)
            # bigdl-llm changes end
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
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

    next_cache = None
    from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicCompressCache
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if not isinstance(next_decoder_cache, (DynamicFp8Cache, DynamicCompressCache))
            else next_decoder_cache
        )

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                     if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def llama_model_forward_4_38_internal(
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
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else \
        self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if (input_ids is None) ^ (inputs_embeds is not None):
        invalidInputError(False,
                          f"You cannot specify both input_ids and inputs_embeds at the same time,"
                          f" and must specify either one")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. "
            "Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            # bigdl-llm changes:
            curr_device = decoder_layer.input_layernorm.weight.device
            if causal_mask is not None:
                causal_mask = causal_mask.to(curr_device)
            if position_ids is not None:
                position_ids = position_ids.to(curr_device)
            # bigdl-llm changes end
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
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

    next_cache = None
    from ipex_llm.transformers.kv import DynamicFp8Cache, DynamicCompressCache
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if not isinstance(next_decoder_cache, (DynamicFp8Cache, DynamicCompressCache))
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states,
                                 all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def llama_model_forward_4_36_internal(
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
    output_attentions = output_attentions if output_attentions is not None else \
        self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False,
                          "You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        invalidInputError(False, "You have to specify either input_ids or inputs_embeds")

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
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) \
            else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        from transformers.models.llama.modeling_llama import \
            _prepare_4d_causal_attention_mask_for_sdpa
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing."
                " Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
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
                use_cache,
            )
        else:
            # bigdl-llm changes:
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
                use_cache=use_cache,
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

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache \
            else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache,
                                 all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def llama_model_forward(
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
    output_attentions = output_attentions if output_attentions is not None \
        else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False,
                          "You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        invalidInputError(False, "You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

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
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing."
                " Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions,
                                  padding_mask=padding_mask)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
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
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache,
                                 all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

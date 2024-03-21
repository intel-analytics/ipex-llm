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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
#
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
""" PyTorch Mistral model."""
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mistral.modeling_mistral import MistralModel
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb, \
    apply_rotary_pos_emb_no_cache_xpu
from bigdl.llm.transformers.models.utils import is_enough_kv_cache_room_4_31, \
    is_enough_kv_cache_room_4_36
from bigdl.llm.transformers.low_bit_linear import SYM_INT4, FP8E5, IQ2_XXS
from bigdl.llm.transformers.models.utils import use_flash_attention, use_esimd_sdp
try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = Tuple[torch.Tensor]
KV_CACHE_ALLOC_BLOCK_LENGTH = 256


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


def should_use_fuse_rope(self, hidden_states, position_ids):
    use_fuse_rope = hidden_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and hidden_states.requires_grad)
    use_fuse_rope = use_fuse_rope and position_ids is not None
    return use_fuse_rope


def use_decoding_fast_path(q_type, use_fuse_rope, enough_kv_room, bs):
    return q_type in [SYM_INT4, FP8E5, IQ2_XXS] and \
        use_fuse_rope and enough_kv_room and bs == 1


def compute_attn_outputs_weights(query_states, key_states, value_states, bsz, q_len, kv_seq_len,
                                 num_heads, head_dim, hidden_size, attention_mask):
    attn_weights = torch.matmul(
        query_states.to(key_states.dtype),
        key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        invalidInputError(
            False,
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)},"
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
        softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states.to(query_states.dtype))

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        invalidInputError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)},"
            f" but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    return attn_output, attn_weights


def mistral_model_forward_4_36(
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
    from bigdl.llm.transformers.kv import DynamicFp8Cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if use_cache and use_quantize_kv_cache(self.layers[0].mlp.up_proj, input_ids):
        if not isinstance(past_key_values, DynamicFp8Cache):
            past_key_values = DynamicFp8Cache.from_legacy_cache(past_key_values)
    return MistralModel.forward(
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


def mistral_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    padding_mask: Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache(self.q_proj, hidden_states):
        forward_function = mistral_attention_forward_quantized
    else:
        forward_function = mistral_attention_forward_original
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask
    )


def mistral_attention_forward_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    padding_mask: Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value)
    decoding_fast_path = use_decoding_fast_path(self.q_proj.qtype,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len)

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
        import linear_q4_0
        query_states, key_states, value_states = linear_q4_0.forward_qkv(hidden_states,
                                                                         self.q_proj.weight,
                                                                         self.k_proj.weight,
                                                                         self.v_proj.weight,
                                                                         position_ids,
                                                                         tmp_cache_k, tmp_cache_v,
                                                                         self.q_proj.weight.qtype,
                                                                         self.v_proj.weight.qtype,
                                                                         0,
                                                                         self.head_dim)
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
            kv_seq_len += past_key_value[0].shape[-2]

        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "mistral")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "mistral")

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
    kv_seq_len = key_states.shape[-2]
    if past_key_value is None:
        attn_weights = torch.matmul(query_states.to(key_states.dtype),
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            invalidInputError(
                False,
                f"Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if use_cache:
            k_cache, v_cache = init_fp8_kv_cache(
                bsz, self.num_heads, kv_seq_len, self.head_dim,
                device=query_states.device, new_layout=True
            )
            key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                           key_states, value_states)
            past_key_value = (key_states, value_states)
    else:
        k_cache, v_cache = past_key_value
        key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                       key_states, value_states, new_layout=True)
        kv_seq_len = key_states.shape[-2]
        past_key_value = (key_states, value_states)

        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            attn_weights = attn_weights / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size "
                    f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    invalidInputError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},"
                        f" but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                 dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import linear_q4_0
            attn_output = linear_q4_0.sdp_fp8(query_states, key_states, value_states,
                                              attention_mask)
            attn_weights = None

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def mistral_attention_forward_original(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    padding_mask: Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value)
    decoding_fast_path = use_decoding_fast_path(self.q_proj.qtype,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len)
    decoding_fast_path = decoding_fast_path and not self.q_proj.enable_xetla

    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        kv_seq_len = past_key_value[0].shape[-2]
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        import linear_q4_0
        query_states, key_states, value_states = linear_q4_0.forward_qkv(hidden_states,
                                                                         self.q_proj.weight,
                                                                         self.k_proj.weight,
                                                                         self.v_proj.weight,
                                                                         position_ids,
                                                                         cache_k, cache_v,
                                                                         self.q_proj.weight.qtype,
                                                                         self.v_proj.weight.qtype,
                                                                         kv_seq_len,
                                                                         self.head_dim)
        kv_seq_len += 1
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
            kv_seq_len += past_key_value[0].shape[-2]

        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "mistral")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "mistral")

        if past_key_value is not None:
            # reuse k, v, self_attention
            cache_k = past_key_value[0]
            cache_v = past_key_value[1]
            if not enough_kv_room:
                # allocate new
                new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                           self.num_key_value_heads,  # Support GQA
                                                           self.head_dim,
                                                           cache_k.size(2),
                                                           kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                           dtype=cache_k.dtype,
                                                           device=device)

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
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    elif use_esimd_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
        import linear_fp16_esimd
        attn_output = linear_fp16_esimd.sdp_forward(query_states,
                                                    key_states,
                                                    value_states)
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    else:
        attn_output, attn_weights = compute_attn_outputs_weights(query_states,
                                                                 key_states,
                                                                 value_states,
                                                                 bsz, q_len, kv_seq_len,
                                                                 self.num_heads, self.head_dim,
                                                                 self.hidden_size, attention_mask)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def mistral_attention_forward_4_36(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Cache]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    if use_quantize_kv_cache(self.q_proj, hidden_states):
        forward_function = mistral_attention_forward_4_36_quantized
    else:
        forward_function = mistral_attention_forward_4_36_original
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        kwargs=kwargs
    )


def mistral_attention_forward_4_36_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Cache]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx, seq_len=q_len)
    decoding_fast_path = use_decoding_fast_path(self.q_proj.qtype,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len)

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
        import linear_q4_0
        query_states, key_states, value_states = linear_q4_0.forward_qkv(hidden_states,
                                                                         self.q_proj.weight,
                                                                         self.k_proj.weight,
                                                                         self.v_proj.weight,
                                                                         position_ids,
                                                                         tmp_cache_k, tmp_cache_v,
                                                                         self.q_proj.weight.qtype,
                                                                         self.v_proj.weight.qtype,
                                                                         0,
                                                                         self.head_dim)
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
                invalidInputError(
                    False,
                    f"The cache structure has changed since version v4.36. "
                    "If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, "
                    "please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "mistral")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "mistral")

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
    kv_seq_len = key_states.shape[-2]
    if len(past_key_value.key_cache) <= self.layer_idx:
        attn_weights = torch.matmul(query_states.to(key_states.dtype),
                                    key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            invalidInputError(
                False,
                f"Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if use_cache:
            cache_kwargs = None
            key_states, value_states = past_key_value.update(key_states, value_states,
                                                             self.layer_idx, cache_kwargs,
                                                             new_layout=True)
    else:
        cache_kwargs = None  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states,
                                                         self.layer_idx, cache_kwargs,
                                                         new_layout=True)
        kv_seq_len = key_states.shape[-2]
        if query_states.size(2) != 1 or query_states.device.type != 'xpu':
            key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                            query_states.dtype)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            attn_weights = attn_weights / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                invalidInputError(
                    False,
                    f"Attention weights should be of size "
                    f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    invalidInputError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},"
                        f" but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                 dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
        else:
            import linear_q4_0
            attn_output = linear_q4_0.sdp_fp8(query_states, key_states, value_states,
                                              attention_mask)
            attn_weights = None

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def mistral_attention_forward_4_36_original(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Cache]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    bsz, q_len, hidden_size = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx)
    decoding_fast_path = use_decoding_fast_path(self.q_proj.qtype,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len)
    decoding_fast_path = decoding_fast_path and not self.q_proj.enable_xetla

    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)

        cache_k = past_key_value.key_cache[self.layer_idx]
        cache_v = past_key_value.value_cache[self.layer_idx]

        kv_seq_len = cache_k.shape[-2]

        import linear_q4_0
        query_states, key_states, value_states = linear_q4_0.forward_qkv(hidden_states,
                                                                         self.q_proj.weight,
                                                                         self.k_proj.weight,
                                                                         self.v_proj.weight,
                                                                         position_ids,
                                                                         cache_k, cache_v,
                                                                         self.q_proj.weight.qtype,
                                                                         self.v_proj.weight.qtype,
                                                                         kv_seq_len,
                                                                         self.head_dim)
        kv_seq_len += 1

        # update past_key_value's seem_tokens and kv caches.
        if self.layer_idx == 0:
            past_key_value.seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states

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
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "mistral")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "mistral")

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

                key_states, value_states = append_kv_cache(cache_k, cache_v,
                                                           key_states, value_states)

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
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    else:
        attn_output, attn_weights = compute_attn_outputs_weights(query_states,
                                                                 key_states,
                                                                 value_states,
                                                                 bsz, q_len, kv_seq_len,
                                                                 self.num_heads,
                                                                 self.head_dim,
                                                                 self.hidden_size,
                                                                 attention_mask)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value

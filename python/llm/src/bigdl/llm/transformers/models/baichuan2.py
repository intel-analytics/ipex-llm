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

# This file is adapted from
# https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/cb7fc748b78b7ea99772e4cf76db155729ce774e/modeling_baichuan.py
# and
# https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/c6f8592a60b4ad73c210b28dd2ab3cca51abbf93/modeling_baichuan.py

import math
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch.nn import functional as F
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.transformers.models.utils import init_fp8_kv_cache, append_fp8_kv_cache, \
    restore_fp8_kv_cache, use_quantize_kv_cache
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, \
    append_kv_cache, is_enough_kv_cache_room_4_31
from bigdl.llm.transformers.models.utils import use_flash_attention, use_esimd_sdp
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb, SILU
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu
from bigdl.llm.transformers.models.utils import mlp_fusion_check
from transformers.utils import logging
logger = logging.get_logger(__name__)

try:
    from xformers import ops as xops
except ImportError:
    xops = None
    logger.warning(
        "Xformers is not installed correctly. If you want to use memory_efficient_attention to "
        "accelerate training use the following command to install Xformers\npip install xformers."
    )


KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def baichuan_13b_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu" and not (self.training and hidden_states.requires_grad):
        import linear_q4_0
        x_2d = hidden_states.reshape(-1, hidden_states.size(-1)).contiguous()
        output = linear_q4_0.rms_norm(self.weight, x_2d, self.epsilon)
        if 1 < x_2d.size(0) <= 64:   # may use XMX, need copy
            output = output.clone()
        return output.reshape(hidden_states.shape)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)
    return self.weight * hidden_states.to(input_dtype)


def baichuan_mlp_forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    qtype = getattr(self.gate_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training) and not self.down_proj.enable_xetla:
        import linear_q4_0
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        return self.down_proj(linear_q4_0.mlp_forward_xpu(
            x_2d, self.gate_proj.weight.data, self.up_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.gate_proj.out_len,
            SILU, qtype
        ))
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def baichuan_attention_forward_7b(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache(self.W_pack, hidden_states):
        forward_function = baichuan_attention_forward_7b_quantized
    else:
        forward_function = baichuan_attention_forward_7b_origin
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache
    )


def baichuan_attention_forward_7b_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = torch.chunk(proj, 3, -1)
    # batch_size x source_len x hidden_size
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x target_len x head_size
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x source_len x hidden_size
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    if query_states.device.type == "xpu" and not (self.training and query_states.requires_grad):
        query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                     key_states,
                                                                     position_ids,
                                                                     "baichuan")
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids, "baichuan")
    if past_key_value is None:
        kv_seq_len = key_states.shape[-2]
        k_cache, v_cache = init_fp8_kv_cache(
            bsz, self.num_heads, kv_seq_len, self.head_dim,
            device=device, new_layout=True
        )
    else:
        k_cache, v_cache = past_key_value
    key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                   key_states, value_states, new_layout=True)

    past_key_value = (key_states, value_states) if use_cache else None

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attention_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))

    scaling_factor = 1 / math.sqrt(query_states.size(-1))
    if query_states.size(2) != 1 or device.type != 'xpu':
        key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                        query_states.dtype)
        attn_output = torch.matmul(query_states * scaling_factor, key_states.transpose(-2, -1))

        if attention_mask is not None:
            attn_output += attention_mask
        attn_output = torch.softmax(attn_output, -1)
        attn_output = attn_output.to(hidden_states.dtype)
        attn_output = torch.matmul(attn_output, value_states)
    else:
        import linear_q4_0
        attn_output = linear_q4_0.sdp_fp8(query_states, key_states, value_states,
                                          attention_mask)
        attn_weights = None
    attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def baichuan_attention_forward_7b_origin(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = torch.chunk(proj, 3, -1)
    # batch_size x source_len x hidden_size
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x target_len x head_size
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # batch_size x source_len x hidden_size
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    enough_kv_room = True
    if past_key_value is not None:
        enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value, seq_len=kv_seq_len)
        kv_seq_len += past_key_value[0].shape[-2]
    if query_states.device.type == "xpu" and not (self.training and query_states.requires_grad):
        query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                     key_states,
                                                                     position_ids,
                                                                     "baichuan")
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids, "baichuan")
    # [bsz, nh, t, hd]

    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if past_key_value is not None:
        # reuse k, v, self_attention
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if not enough_kv_room:
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,
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
                                                         self.num_heads,
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

    if xops is not None and self.training:
        attn_weights = None
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = xops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask()
        )
    else:
        if not self.training and not hidden_states.requires_grad and \
                use_flash_attention(query_states, key_states, attention_mask):
            attn_output = F.scaled_dot_product_attention(query_states.to(dtype=torch.float16),
                                                         key_states.to(dtype=torch.float16),
                                                         value_states.to(dtype=torch.float16),
                                                         is_causal=True)
            attn_weights = None
        elif not self.training and not hidden_states.requires_grad and \
                use_esimd_sdp(q_len, key_states.shape[2], self.head_dim, query_states):
            import linear_fp16_esimd
            attn_output = linear_fp16_esimd.sdp_forward(query_states,
                                                        key_states,
                                                        value_states)
            attn_output = attn_output.view(query_states.shape)
            attn_weights = None
        else:
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attention_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))

            scaling_factor = 1 / math.sqrt(query_states.size(-1))
            attn_output = torch.matmul(query_states * scaling_factor, key_states.transpose(-2, -1))
            if attention_mask is not None:
                attn_output += attention_mask
            attn_output = torch.softmax(attn_output, -1)
            attn_output = torch.matmul(attn_output, value_states)

        attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(hidden_states.dtype), attn_weights, past_key_value


def baichuan_attention_forward_13b(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_quantize_kv_cache(self.W_pack, hidden_states):
        forward_function = baichuan_attention_forward_13b_quantized
    else:
        forward_function = baichuan_attention_forward_13b_origin
    return forward_function(
        self=self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )


def baichuan_attention_forward_13b_quantized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = (
        proj.unflatten(-1, (3, self.hidden_size))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
    )
    query_states = (
        proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    key_states = (
        proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    value_states = (
        proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )

    if past_key_value is None:
        kv_seq_len = key_states.shape[-2]
        k_cache, v_cache = init_fp8_kv_cache(
            bsz, self.num_heads, kv_seq_len, self.head_dim,
            device=device
        )
    else:
        k_cache, v_cache = past_key_value
    key_states, value_states = append_fp8_kv_cache(k_cache, v_cache,
                                                   key_states, value_states)
    past_key_value = (key_states, value_states)

    if query_states.size(2) != 1 or device.type != 'xpu':
        key_states, value_states = restore_fp8_kv_cache(key_states, value_states,
                                                        query_states.dtype)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    else:
        import linear_q4_0
        attn_weights = linear_q4_0.query_key_fp8_matmul(query_states, key_states)

    attn_weights = attn_weights / math.sqrt(self.head_dim)

    if attention_mask is not None:
        if q_len == 1:  # inference with cache
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, -1:, :]
            else:
                attention_mask = attention_mask[:, -1:, :]
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights,
                                 torch.tensor(torch.finfo(attn_weights.dtype).min))

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(hidden_states.dtype)

    if query_states.size(2) != 1 or device.type != 'xpu':
        attn_output = torch.matmul(attn_weights, value_states)
    else:
        import linear_q4_0
        attn_output = linear_q4_0.attn_value_fp8_matmul(attn_weights,
                                                        value_states.transpose(-1, -2))

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def baichuan_attention_forward_13b_origin(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    proj = self.W_pack(hidden_states)
    proj = (
        proj.unflatten(-1, (3, self.hidden_size))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
    )
    query_states = (
        proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    key_states = (
        proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )
    value_states = (
        proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    enough_kv_room = True
    if past_key_value is not None:
        enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value, seq_len=kv_seq_len)
        kv_seq_len += past_key_value[0].shape[-2]

    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if past_key_value is not None:
        # reuse k, v, self_attention
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        if not enough_kv_room:
            if device.type == 'xpu':
                torch.xpu.empty_cache()
            # allocate new
            new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                       self.num_heads,
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
                                                         self.num_heads,
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
    if xops is not None and self.training:
        attn_weights = None
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)
        # attn_output = xops.memory_efficient_attention(
        #     query_states, key_states, value_states, attn_bias=attention_mask
        # )
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True,
                                            enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                         attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2)
    else:
        attn_weights = torch.matmul(
            query_states.to(dtype=key_states.dtype), key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            if attention_mask.shape[-2] == attn_weights.shape[-2]:
                attn_weights = attn_weights + attention_mask
            else:
                # support for Baichuan/Baichuan2 13B Chat running speculative decoding
                # split attention mask on dim -2
                split_sizes = [attention_mask.shape[-2] - attn_weights.shape[-2],
                               attn_weights.shape[-2]]
                # the last chunk of splited is the new attention mask
                attention_mask = attention_mask.split(split_sizes, dim=-2)[-1]
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights.to(dtype=value_states.dtype), value_states)

        attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[: tensor.shape[0] * attn_heads, :maxpos, :maxpos]


def baichuan_13b_gen_alibi_mask(tensor, n_head, max_pos):
    # May use fp16 for alibi mask to further reduce memory
    slopes = torch.Tensor(_get_interleave(n_head))  # .half()
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)  # .half()
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    if tensor.device.type == "xpu":
        alibi_mask = alibi_mask.to(tensor.device)
    return alibi_mask


MASK_BLOCK_SIZE = 512


def baichuan_13b_get_alibi_mask(self, tensor, seq_length_with_past):
    if self.training:
        slopes = torch.Tensor(_get_interleave(self.n_head))
        position_point = (
            torch.arange(seq_length_with_past) - seq_length_with_past + 1
        )
        position_point = (
            position_point.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.n_head, seq_length_with_past, -1)
        )
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(
            -1, -2
        )
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
        mask = _buffered_future_mask(
            tensor, seq_length_with_past, alibi, self.n_head
        )
    else:
        if self.first_run:
            # Override the default max_cache_pos=4096 for memory considerations
            self.max_cache_pos = seq_length_with_past + MASK_BLOCK_SIZE
            self.first_run = False
            self.register_buffer(
                "future_mask",
                baichuan_13b_gen_alibi_mask(tensor, self.n_head, self.max_cache_pos),
                persistent=False,
            )
        if seq_length_with_past > self.max_cache_pos:
            # When max_cache_pos is not enough for current sequence length,
            # increase by MASK_BLOCK_SIZE and recalculate future_mask.
            self.max_cache_pos = seq_length_with_past + MASK_BLOCK_SIZE
            self.register_buffer(
                "future_mask",
                baichuan_13b_gen_alibi_mask(tensor, self.n_head, self.max_cache_pos),
                persistent=False,
            )
        mask = self.future_mask[
            : self.n_head, :seq_length_with_past, :seq_length_with_past
        ]
    return mask

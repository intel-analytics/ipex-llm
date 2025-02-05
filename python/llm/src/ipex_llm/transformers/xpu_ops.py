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

import torch
import xe_linear
import xe_batch
import xe_addons


@torch.library.register_fake("ipex_llm::forward_new")
def _(x, weight, qtype, output_size):
    return torch.empty([x.size(0), output_size],
                       dtype=x.dtype, device=x.device)


# @torch.library.register_fake("ipex_llm::dequant")
# def _(x, weight, qtype):
#     return ???


@torch.library.register_fake("ipex_llm::mlp_forward_xpu")
def _(x, weight1, weight2, batch_size, state_size, output_size, act_type, qtype):
    return torch.empty([batch_size, output_size],
                       dtype=x.dtype, device=x.device)


@torch.library.register_fake("ipex_llm::rwkv_linear_attention_v4")
def _(time_decay, time_first, key, value, num_state, den_state, max_state):
    return torch.empty_like(key)


@torch.library.register_fake("ipex_llm::rwkv_linear_attention_v5")
def _(time_decay, time_first, receptance, key, value, state):
    bsz, n_heads, seq_len, head_dim = key.shape
    return torch.empty([bsz, seq_len, n_heads, head_dim],
                       dtype=key.dtype, device=key.device)


@torch.library.register_fake("ipex_llm::rwkv_time_shift")
def _(hidden, shifted, mix):
    bsz, seq_len, hidden_size = hidden.shape
    return torch.empty([mix.size(0), bsz, seq_len, hidden_size],
                       dtype=hidden.dtype, device=hidden.device)


@torch.library.register_fake("ipex_llm::dequantize_rows")
def _(x, weight, qtype, state_size, output_size):
    return torch.empty([x.size(0), x.size(1), state_size],
                       dtype=torch.float, device=weight.device)


# @torch.library.register_fake("ipex_llm::batch_forward")
# def _(x, weight, qtype):
#     return ???


@torch.library.register_fake("ipex_llm::sdp")
def _(query, key, value, mask):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::sdp_fp8")
def _(query, key, value, mask):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::sdp_causal")
def _(query, key, value, mask, scale):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::sdp_fp8_causal")
def _(query, key, value, mask, scale):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::sdp_non_causal")
def _(query, key, value, mask, scale):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::sdp_fp8_non_causal")
def _(query, key, value, mask, scale):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::siglip_sdp_non_causal")
def _(query, key, value, mask):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::gemma2_sdp")
def _(query, key, value, mask, f1, f2):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::gemma2_sdp_causal")
def _(query, key, value, mask, f1, f2):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.register_fake("ipex_llm::rms_norm")
def _(weight, x, eps):
    return torch.empty_like(x)


@torch.library.register_fake("ipex_llm::layer_norm")
def _(x, weight, bias, eps):
    return torch.empty_like(x)


@torch.library.register_fake("ipex_llm::rotary_half_inplaced")
def _(inv_freq, position_ids, query, key):
    pass


@torch.library.register_fake("ipex_llm::rotary_two_inplaced")
def _(inv_freq, position_ids, query, key):
    pass


@torch.library.register_fake("ipex_llm::rotary_half_with_cache_inplaced")
def _(query, key, cos, sin):
    pass


@torch.library.register_fake("ipex_llm::rotary_two_with_cache_inplaced")
def _(query, key, cos, sin, half_layout):
    pass


@torch.library.register_fake("ipex_llm::mlp_silu_mul_inplaced")
def _(gate, up):
    pass


@torch.library.register_fake("ipex_llm::quantize_key_value")
def _(key, value, key_output, value_output):
    pass


@torch.library.register_fake("ipex_llm::dequantize_key_value")
def _(key, value, key_output, value_output):
    pass


@torch.library.register_fake("ipex_llm::attn_softmax_inplaced")
def _(attn):
    pass

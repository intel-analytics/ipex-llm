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
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/idefics2/modeling_idefics2.py
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
#
# Some parts of this file is adapted from
# https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/resampler.py
# which is licensed under Apache License 2.0:
#
# Copyright 2024 OpenBMB

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ipex_llm.utils.common.log4Error import invalidInputError
from torch import Tensor
import warnings
from torch.nn.functional import *
from torch.nn.modules.activation import *
from intel_npu_acceleration_library.backend.factory import NNFactory
import numpy as np
from functools import partial
import uuid
from ipex_llm.transformers.npu_models.mp_models_base import run_model
from ipex_llm.transformers.npu_models.convert import module_optimization


class MinicpmVConv2d(NNFactory):
    def __init__(
        self,
        input_shape,
        weight_shape,
        bias,
        strides,
        padding,
        dilation,
        groups,
        device: str = "NPU",
    ):
        super().__init__(False, device)

        # define input
        input = self.parameter(input_shape, dtype=np.float16)
        weight = self.parameter(weight_shape, dtype=np.float16)
        if bias is not None:
            bias_node = self.parameter((1, weight_shape[0], 1, 1), dtype=np.float16)
        else:
            bias_node = None

        input = self.concat(input, input, axis=2)  # current workaround for compile error
        res = self.convolution(input_node=input,
                               weights_node=weight,
                               bias=bias_node,
                               strides=strides,
                               padding=padding,
                               dilation=dilation,
                               groups=groups)
        res = self.slice(res, begin=[0, 0, 0, 0],
                         end=[res.shape[0], res.shape[1], 1, res.shape[3]])
        # define outputs
        res = self.convert_to_fp16(res)

        print("start compiling")
        self.compile()


class MinicpmVPatchEmbedding(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        strides=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super().__init__()

        self.op_id = str(uuid.uuid4())
        self.parameters = [weight]
        if bias is not None:
            self.parameters.append(bias)
        self.backend_cls = partial(
            MinicpmVConv2d,
            weight_shape=weight.shape,
            bias=bias,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        x = x.to(torch.float16)
        return run_model(x, self.parameters, self.backend_cls, self.op_id)


class LayerNorm(NNFactory):
    def __init__(
        self,
        input_shape,
        weight_shape,
        bias_shape,
        eps,
        device: str = "NPU",
    ):
        super().__init__(False, device)

        # define input
        input = self.parameter(input_shape, dtype=np.float16)
        weight = self.parameter(weight_shape, dtype=np.float16)
        bias = self.parameter(bias_shape, dtype=np.float16)

        input = self.convert_to_fp32(input)
        mean_res = self.reduce_mean(input, -1, keep_dims=True,)
        variance = self.reduce_mean(
            self.power(input - mean_res, self.constant(np.array([[2]], dtype=np.float32))),
            -1,
            keep_dims=True,
        )
        eps = self.constant(eps)
        input = self.eltwise_div(input - mean_res, self.sqrt(self.eltwise_add(variance, eps)))
        weight = self.convert_to_fp32(weight)
        input = self.eltwise_mul(weight, input)
        bias = self.convert_to_fp32(bias)
        input = self.eltwise_add(bias, input)

        # define outputs
        input = self.convert_to_fp16(input)

        print("start compiling")
        self.compile()


class MinicpmVLayerNorm(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        eps=1e-6,
    ):
        super().__init__()
        self.op_id = str(uuid.uuid4())
        self.parameters = [weight, bias]
        self.backend_cls = partial(
            LayerNorm,
            weight_shape=weight.shape,
            bias_shape=bias.shape,
            eps=eps,
        )

    def forward(self, x):
        x = x.to(torch.float16)
        return run_model(x, self.parameters, self.backend_cls, self.op_id)


@module_optimization
def replace_with_Layernorm(layer, qtype=None, device='NPU',
                           modules_to_not_convert=[], group_size=0, imatrix=None):
    if isinstance(layer, torch.nn.LayerNorm):
        return MinicpmVLayerNorm(
            weight=layer.weight.to(torch.float16),
            bias=layer.bias.to(torch.float16),
        )


def pad_mlp_fc2(module: torch.nn.Module):
    if hasattr(module, 'fc2') and module.fc2.in_features == 4304:
        new_linear = torch.nn.Linear(0, 0, bias=True)
        padded_weight = torch.cat((module.fc2.weight, module.fc2.weight[:, :(1152*4-4304)]), dim=1)
        new_weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        new_linear.weight = new_weight
        new_linear.bias = module.fc2.bias
        new_linear.in_features = new_weight.size(1)
        new_linear.out_features = new_weight.size(0)
        module.fc2 = new_linear
        del new_linear


def pad_mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = F.pad(hidden_states,
                          (0, (1152*4-4304), 0, 0, 0, 0))
    hidden_states = self.fc2(hidden_states)
    return hidden_states


def pad_lm_head(module: torch.nn.Module):
    if hasattr(module, 'lm_head') and module.lm_head.in_features == 3584 \
       and module.lm_head.out_features == 151666:
        new_linear = torch.nn.Linear(0, 0, bias=False)
        padded_weight = F.pad(module.lm_head.weight,
                              (0, 0, 0, 152064-151666))  # 152064 is qwen2-7b vocab_size
        new_weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        new_linear.weight = new_weight
        new_linear.in_features = new_weight.size(1)
        new_linear.out_features = new_weight.size(0)
        module.lm_head = new_linear
        del new_linear


def lm_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self(hidden_states)
    hidden_states = hidden_states[:, :, :151666]
    return hidden_states


def encoder_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    batch_size, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, q_len,
                                     self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len,
                                 self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len,
                                     self.num_heads, self.head_dim).transpose(1, 2)

    k_v_seq_len = key_states.shape[-2]
    # ipex-llm change starts
    attn_weights = torch.matmul(query_states.float(),
                                key_states.float().transpose(2, 3)) * self.scale
    # ipex-llm change ends

    if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
        invalidInputError(False,
                          f"Attention weights should be of size ({batch_size, self.num_heads, }"
                          f"{q_len, k_v_seq_len}), but is {attn_weights.size()}")

    if attention_mask is not None:
        invalidInputError(attention_mask.size() == (batch_size, 1, q_len, k_v_seq_len),
                          f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}"
                          f", but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    # ipex-llm change starts
    attn_output = torch.matmul(attn_weights.float(), value_states.float())
    # ipex-llm change ends

    invalidInputError(attn_output.size() == (batch_size, self.num_heads, q_len, self.head_dim),
                      f"`attn_output` should be of size ({batch_size, self.num_heads, }"
                      f"{q_len, self.head_dim}), but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q.float(), w_q.float(), b_q.float()), \
        linear(k.float(), w_k.float(), b_k.float()), \
        linear(v.float(), w_v.float(), b_v.float())


def multi_head_attn_forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    # port from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/resampler.py#L338
    # to solve conflict of fp16 and fp32 dtype
    is_batched = True if query.dim() == 3 else False

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads

    # compute in-projection
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # prep attention mask
    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            invalidInputError(attn_mask.shape == correct_2d_size,
                              f"The shape of the 2D attn_mask is {attn_mask.shape},"
                              f"but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            invalidInputError(attn_mask.shape == correct_3d_size,
                              f"The shape of the 3D attn_mask is {attn_mask.shape},"
                              f" but should be {correct_3d_size}.")
        else:
            invalidInputError(False, f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # (deep breath) calculate attention and out projection
    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask.float(),
                                                q_scaled.float(), k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights.float(), v.float())

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


def resampler_forward(self, x, tgt_sizes=None):
    # port from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/resampler.py#L130
    bs = x.shape[0]

    device = x.device
    dtype = x.dtype

    patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

    self._adjust_pos_cache(tgt_sizes, device=device)

    max_patch_len = torch.max(patch_len)
    key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

    pos_embed = []
    for i in range(bs):
        tgt_h, tgt_w = tgt_sizes[i]
        pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))
        key_padding_mask[i, patch_len[i]:] = True

    pos_embed = torch.nn.utils.rnn.pad_sequence(
        pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D

    x = self.kv_proj(x)  # B * L * D
    x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

    q = self.ln_q(self.query)  # Q * D

    out = self.attn(
        self._repeat(q, bs),  # Q * B * D
        x + pos_embed,  # L * B * D +  L * B * D
        x,
        key_padding_mask=key_padding_mask)[0]
    #  out: Q * B * D
    x = out.permute(1, 0, 2)  # B * Q * D

    x = self.ln_post(x)
    # ipex-llm change starts
    x = x.float() @ self.proj.float()
    x = x.to(torch.float16)
    # ipex-llm change ends
    return x

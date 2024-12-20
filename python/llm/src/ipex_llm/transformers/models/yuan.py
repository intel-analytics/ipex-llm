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
# https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/7ab7b3c18eb8e5232ce2a3f720d4e6f4b53a2806/yuan_hf_model.py
# which is licensed under Apache License 2.0:
#
# https://huggingface.co/IEITYuan/Yuan2-2B-hf/blob/7ab7b3c18eb8e5232ce2a3f720d4e6f4b53a2806/README.md#%E5%A3%B0%E6%98%8E%E4%B8%8E%E5%8D%8F%E8%AE%AEterms-and-conditions
#

import math
from typing import Optional, Tuple

import torch

from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb, \
    mlp_fusion_check, fp16_fusion_check
from ipex_llm.transformers.models.utils import use_quantize_kv_cache
from ipex_llm.transformers.models.utils import SILU, update_past_key_value
from ipex_llm.transformers.models.utils import should_use_fuse_rope


def merge_qk(module: torch.nn.Module):
    if "YuanAttention" in module.__class__.__name__:
        q_weight = module.q_proj.weight.data
        k_weight = module.k_proj.weight.data
        num_heads = module.num_heads
        head_dim = module.head_dim
        hidden_size = module.hidden_size

        merged_qk_proj = torch.nn.Linear(0, 0, False)
        weight = torch.cat([
            q_weight.view(num_heads, head_dim, hidden_size)[0::2, :, :],
            k_weight.view(num_heads, head_dim, hidden_size)[0::2, :, :],
            q_weight.view(num_heads, head_dim, hidden_size)[1::2, :, :],
            k_weight.view(num_heads, head_dim, hidden_size)[1::2, :, :],
        ], dim=0).view(num_heads * head_dim * 2, hidden_size)
        merged_qk_proj.weight = torch.nn.Parameter(weight, requires_grad=False)
        merged_qk_proj.in_features = hidden_size
        merged_qk_proj.out_features = num_heads * head_dim * 2
        module.qk_proj = merged_qk_proj

        del module.q_proj
        del module.k_proj


def yuan_localized_filtering_forward(
    self,
    inputs: torch.Tensor,
    before_hidden_states: torch.Tensor,
    dtype: torch.dtype,
):
    if self.conv1.weight.dtype != torch.half:
        self.half()

    invalidInputError(self.lf_conv2d_num_pad == 1, "padding must be 1")
    invalidInputError(not self.training, ("training is not supported for now, "
                                          "please call model.eval() before inference"))
    if before_hidden_states is None:
        inputs = inputs.half()
        lf_output = self._inference_forward(inputs, None)
    else:
        # only change next token logic
        bsz, seq_len, embed_dim = inputs.size()
        seq_len_before, _, _ = before_hidden_states.size()
        invalidInputError(seq_len == 1 and seq_len_before == 3,
                          f"wrong sequence length: {seq_len} {seq_len_before}")

        residual = before_hidden_states[-1:, :, :]
        inputs = before_hidden_states.view(3, 1, bsz, embed_dim).permute(2, 3, 0, 1)

        output1 = self.conv1(inputs)
        output2 = self.conv2(output1[:, :, 1:-1, :])
        output2 = output2[:, :, 1:-1, :]
        output2 = output2.view(1, bsz, embed_dim)

        invalidInputError(output2.shape == residual.shape,
                          f"wrong shape: {output2.shape} {residual.shape}")

        lf_output = self.output_layernorm(output2 + residual)
        lf_output = lf_output.transpose(0, 1)

    lf_output = lf_output.to(dtype)
    return lf_output


def yuan_mlp_forward(
    self,
    x: torch.Tensor,
    residual=None
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    bsz, hidden_size = x_2d.shape
    qtype = getattr(self.up_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, self.training):
        import xe_linear
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out = self.down_proj(xe_linear.mlp_forward_xpu(
            x_2d, self.up_proj.weight.data, self.gate_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.up_proj.out_len,
            SILU, qtype
        ))
        if residual is not None:
            return out + residual
        else:
            return out
    elif fp16_fusion_check(self.up_proj, x, self.training) and \
            hidden_size == 4096 and bsz == 1:
        hidden_states1 = torch.ops.torch_ipex.mm_silu(x, self.up_proj.weight)
        hidden_states = torch.ops.torch_ipex.mm_resmul(
            x, self.gate_proj.weight, hidden_states1
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
    else:
        out = self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))
        if residual is not None:
            return out + residual
        else:
            return out


def yuan_attention_forward(
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

    invalidInputError(use_cache, "use_cache=True is needed")
    invalidInputError(not self.use_shareqk, "use_shareqk is not supported for now")

    if past_key_value is None:
        if q_len >= 2:
            before_hidden_states = hidden_states[:, -2:, :].transpose(0, 1).half()
        else:
            before_hidden_states = torch.zeros(2, bsz, self.hidden_size,
                                               dtype=torch.half, device=hidden_states.device)
            before_hidden_states[-1:, :, :] = hidden_states[:, -1:, :].transpose(0, 1)
    else:
        before_hidden_states = past_key_value[2]
        this_hidden_states = torch.cat([
            before_hidden_states,
            hidden_states.transpose(0, 1).half(),
        ], dim=0)
        before_hidden_states = this_hidden_states[-2:, :, ]

    value_states = self.v_proj(hidden_states)
    value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if past_key_value is None:
        hidden_states = yuan_localized_filtering_forward(self.lf_gate, hidden_states,
                                                         None, hidden_states.dtype)
    else:
        hidden_states = yuan_localized_filtering_forward(self.lf_gate, hidden_states,
                                                         this_hidden_states, hidden_states.dtype)

    qk_states = self.qk_proj(hidden_states)
    qk_states = qk_states.view(bsz, q_len, self.num_heads * 2, self.head_dim)
    qk_states = qk_states.transpose(1, 2)
    query_states, key_states = torch.chunk(qk_states, 2, dim=1)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if should_use_fuse_rope(hidden_states, position_ids, self.training):
        import xe_addons
        xe_addons.rotary_half_inplaced(self.rotary_emb.inv_freq, position_ids,
                                       query_states, key_states)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states,
                                                        cos, sin,
                                                        position_ids,
                                                        "yuan")

    # IPEX-LLM OPT: kv cache and quantzie kv cache
    use_quantize_kv = use_quantize_kv_cache(self.qk_proj, hidden_states)
    key_states, value_states = update_past_key_value(
        None if past_key_value is None else (past_key_value[0], past_key_value[1]),
        key_states, value_states,
        kv_seq_len, use_quantize_kv, device
    )
    past_key_value = (key_states, value_states, before_hidden_states) if use_cache else None

    # IPEX-LLM OPT: sdpa
    attn_weights = None
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == kv_seq_len
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

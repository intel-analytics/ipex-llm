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


import math
import torch
from typing import List


def merge_linear(linears: List[torch.nn.Linear]) -> torch.nn.Linear:
    if hasattr(linears[0], "weight"):
        # For GPTQ model, it might be qweight
        new_weight = torch.cat(list(linear.weight.data for linear in linears), dim=0)
        if linears[0].bias is not None:
            new_linear = torch.nn.Linear(0, 0, bias=True)
            new_bias = torch.cat(list(linear.bias.data for linear in linears), dim=0)
            new_linear.bias = torch.nn.Parameter(new_bias, requires_grad=False)
        else:
            new_linear = torch.nn.Linear(0, 0, bias=False)
        new_linear.weight = torch.nn.Parameter(new_weight, requires_grad=False)
        new_linear.in_features = new_weight.size(1)
        new_linear.out_features = new_weight.size(0)
        return new_linear
    else:
        return None


def merge_qkv_base(module: torch.nn.Module, attention_class):
    if (
        isinstance(attention_class, str) and module.__class__.__name__ == attention_class
        or not isinstance(attention_class, str) and isinstance(module, attention_class)
    ):
        qkv_proj = merge_linear([
            module.q_proj,
            module.k_proj,
            module.v_proj,
        ])
        if qkv_proj is not None:
            module.qkv_proj = qkv_proj
            del module.q_proj, module.k_proj, module.v_proj


def padding_linear_hd(linear: torch.nn.Linear,
                      old_head_dim: int, new_head_dim: int) -> torch.nn.Linear:
    in_features, out_features = linear.in_features, linear.out_features

    weight = linear.weight.data
    weight = weight.view(-1, old_head_dim, in_features)
    new_weight = torch.empty([weight.size(0), new_head_dim, in_features],
                             dtype=weight.dtype, device=weight.device)
    new_weight[:, :old_head_dim, :] = weight
    new_weight[:, old_head_dim:, :] = 0
    new_weight = new_weight.view(-1, in_features)
    if linear.bias is not None:
        bias = linear.bias.data
        bias = bias.view(-1, old_head_dim)
        new_bias = torch.empty([bias.size(0), new_head_dim],
                               dtype=bias.dtype, device=bias.device)
        new_bias[:, :old_head_dim] = bias
        new_bias[:, old_head_dim:] = 0
        new_bias = new_bias.flatten()

        new_linear = torch.nn.Linear(0, 0, bias=True)
        new_linear.bias = torch.nn.Parameter(new_bias, requires_grad=False)
    else:
        new_linear = torch.nn.Linear(0, 0, bias=False)
    new_linear.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    new_linear.in_features = new_weight.size(1)
    new_linear.out_features = new_weight.size(0)
    return new_linear


def padding_attention_hd_base(module: torch.nn.Module, attention_class,
                              old_head_dim: int, new_head_dim: int):
    if (
        isinstance(attention_class, str) and module.__class__.__name__ == attention_class
        or not isinstance(attention_class, str) and isinstance(module, attention_class)
    ) and module.head_dim == old_head_dim:
        module.q_proj = padding_linear_hd(module.q_proj, old_head_dim, new_head_dim)
        module.k_proj = padding_linear_hd(module.k_proj, old_head_dim, new_head_dim)
        module.v_proj = padding_linear_hd(module.v_proj, old_head_dim, new_head_dim)
        module.head_dim = new_head_dim
        module.old_head_dim = old_head_dim


def padding_states_hd(states: torch.Tensor, old_head_dim: int, new_head_dim: int):
    bsz, num_heads, seq_len, head_dim = states.size()
    if head_dim == old_head_dim and old_head_dim < new_head_dim:
        new_states = torch.empty([bsz, num_heads, seq_len, new_head_dim],
                                 dtype=states.dtype, device=states.device)
        new_states[:, :, :, :old_head_dim] = states
        new_states[:, :, :, old_head_dim:] = 0
        return new_states
    return states


def padding_qkv_hd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   old_head_dim: int, new_head_dim: int):
    return (
        padding_states_hd(q, old_head_dim, new_head_dim),
        padding_states_hd(k, old_head_dim, new_head_dim),
        padding_states_hd(v, old_head_dim, new_head_dim),
    )


def fuse_mlp_base(module: torch.nn.Module, act: int, x: torch.Tensor):
    from ipex_llm.transformers.models.utils import mlp_fusion_check
    qtype = getattr(module.gate_proj, "qtype", None)
    if mlp_fusion_check(x, qtype, module.training):
        import xe_linear
        x_2d = x.contiguous().view(-1, x.size(-1))
        output = module.down_proj(
            xe_linear.mlp_forward_xpu(
                x_2d, module.gate_proj.weight.data, module.up_proj.weight.data,
                x_2d.size(0), x_2d.size(1), module.gate_proj.out_len,
                act, qtype
            )
        )
        return output.view(x.shape)
    else:
        return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))


def mlp_silu_forward(self, x: torch.Tensor):
    from ipex_llm.transformers.models.utils import SILU
    return fuse_mlp_base(self, SILU, x)


def mlp_gelu_forward(self, x: torch.Tensor):
    from ipex_llm.transformers.models.utils import GELU
    return fuse_mlp_base(self, GELU, x)


def attention_softmax(attn_weights: torch.Tensor):
    if attn_weights.is_contiguous() and attn_weights.device.type == "xpu":
        import xe_addons
        xe_addons.attn_softmax_inplaced(attn_weights)
    else:
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                   dtype=torch.float32).to(attn_weights.dtype)
    return attn_weights


def rms_norm_forward(self, hidden_states: torch.Tensor):
    weight = self.weight
    if hasattr(self, "variance_epsilon"):
        eps = self.variance_epsilon
    else:
        eps = self.epsilon

    if hidden_states.device.type == 'xpu' and hidden_states.dtype in [torch.float, torch.half]:
        import xe_addons
        x_2d = hidden_states.reshape(-1, hidden_states.size(-1)).contiguous()
        output = xe_addons.rms_norm(weight, x_2d, eps)
        return output.reshape(hidden_states.shape)
    else:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states.to(input_dtype)


def layer_norm_forward(self, hidden_states: torch.Tensor):
    if hidden_states.device.type == 'xpu' and hidden_states.dtype in [torch.float, torch.half]:
        import xe_addons
        hidden_size = math.prod(self.normalized_shape)
        x_2d = hidden_states.reshape(-1, hidden_size).contiguous()
        output = xe_addons.layer_norm(x_2d, self.weight, self.bias, self.eps)
        return output.reshape(hidden_states.shape)
    else:
        return torch.nn.functional.layer_norm(
            hidden_states, self.normalized_shape,
            self.weight, self.bias, self.eps
        )


def prepare_mask(mask, bsz, n_heads, seq_length, kv_length, is_causal, dtype, device):
    max_kvs = 128
    padding_kv_length = (kv_length + max_kvs - 1) // max_kvs * max_kvs
    if mask is None:
        if is_causal:
            mask = torch.full([1, 1, seq_length, padding_kv_length], torch.finfo(dtype).min,
                              dtype=dtype, device=device)
            mask.triu_(1)
            mask = mask.expand([bsz, n_heads, seq_length, padding_kv_length])
        elif seq_length != kv_length and seq_length <= 32:
            mask = None
        else:
            mask = torch.zeros([1, 1, 1, padding_kv_length], dtype=dtype, device=device)
            mask[..., kv_length:padding_kv_length] = torch.finfo(dtype).min
            mask = mask.expand([bsz, n_heads, seq_length, padding_kv_length])
    else:
        if seq_length != kv_length and seq_length <= 32:
            mask = mask[..., :seq_length, :kv_length]
            mask = mask.expand([bsz, n_heads, seq_length, kv_length])
        elif mask.size(3) != padding_kv_length:
            new_mask = torch.empty([bsz, 1, seq_length, padding_kv_length],
                                   dtype=dtype, device=device)
            new_mask[:, :, :, :kv_length] = mask[:, 0:1, :seq_length, :kv_length]
            new_mask[:, :, :, kv_length:] = torch.finfo(dtype).min
            new_mask = new_mask.expand([bsz, n_heads, seq_length, padding_kv_length])
            mask.set_(new_mask)     # modify `mask` inplaced
        else:
            mask = mask.expand([bsz, n_heads, seq_length, padding_kv_length])
    return mask


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor,
                                 value: torch.Tensor, mask: torch.Tensor = None,
                                 is_causal: bool = False, scale: float = None) -> torch.Tensor:
    bsz, n_heads, seq_length, head_dim = query.shape
    _, n_kv_heads, kv_length, _ = key.shape

    dtype, device = query.dtype, query.device

    if (
        device.type == "xpu"
        and dtype in [torch.float, torch.half]
        and head_dim in [64, 80, 96, 128]
    ):
        # prepare scale
        scale = 1 / math.sqrt(head_dim) if scale is None else scale

        # prepare mask
        mask = prepare_mask(mask, bsz, n_heads, seq_length, kv_length, is_causal, dtype, device)

        # compute
        import xe_addons
        if is_causal:
            if key.dtype == torch.uint8:
                attn_output = xe_addons.sdp_fp8_causal(query, key, value, mask, scale)
            else:
                attn_output = xe_addons.sdp_causal(query, key, value, mask, scale)
        elif seq_length != kv_length and seq_length <= 32:
            # todo: add scale support
            if key.dtype == torch.uint8:
                attn_output = xe_addons.sdp_fp8(query, key, value, mask)
            else:
                attn_output = xe_addons.sdp(query, key, value, mask)
        else:
            if key.dtype == torch.uint8:
                attn_output = xe_addons.sdp_fp8_non_causal(query, key, value, mask, scale)
            else:
                attn_output = xe_addons.sdp_non_causal(query, key, value, mask, scale)

        return attn_output
    else:
        mask = mask[..., :seq_length, :kv_length] if mask is not None else None

        from ipex_llm.transformers.models.utils import repeat_kv
        if n_heads != n_kv_heads:
            key = repeat_kv(key, n_heads // n_kv_heads)
            value = repeat_kv(value, n_heads // n_kv_heads)

        if is_causal and mask is None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=is_causal, scale=scale
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, mask, scale=scale
            )
        attn_output = attn_output.to(dtype)    # workaround ipex 2.1's bug
        return attn_output

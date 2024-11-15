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
    x_2d = x.view(-1, x.size(-1))
    qtype = getattr(module.gate_proj, "qtype", None)
    if mlp_fusion_check(x_2d, qtype, module.training):
        import xe_linear
        x_2d = x_2d.contiguous()
        return module.down_proj(
            xe_linear.mlp_forward_xpu(
                x_2d, module.gate_proj.weight.data, module.up_proj.weight.data,
                x_2d.size(0), x_2d.size(1), module.gate_proj.out_len,
                act, qtype
            )
        )
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

    if hidden_states.device.type == 'xpu':
        import xe_addons
        x_2d = hidden_states.reshape(-1, hidden_states.size(-1)).contiguous()
        output = xe_addons.rms_norm(weight, x_2d, eps)
        return output.reshape(hidden_states.shape)
    else:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states.to(input_dtype)

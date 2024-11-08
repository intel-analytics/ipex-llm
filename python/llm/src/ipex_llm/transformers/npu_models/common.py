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
from ipex_llm.utils.common.log4Error import invalidInputError


def merge_linear(linears: List[torch.nn.Linear]) -> torch.nn.Linear:
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


def reshape_lm_head_input(x):
    if x.dim() > 3:
        x = x.reshape([-1, x.shape[-2], x.shape[-1]])
    shape = list(x.size())
    if shape[1] > 10:
        shape[1] = 1
        x = x[:, -1, :].view(shape)
    return x


def split_linear(module, module_name, n_splits=2, load=False):
    in_features = module.in_features
    invalidInputError(in_features % n_splits == 0,
                      f"in_features of the linear layer {module_name} must be divisible by"
                      f" n_splits, but got in_features: {in_features}, n_splits: {n_splits}")
    weight_split = torch.tensor_split(module.weight, n_splits, dim=1)
    linear_list = torch.nn.ModuleList()
    bias = module.bias
    from transformers.utils.generic import ContextManagers
    init_contexts = []
    if load:
        from transformers.modeling_utils import no_init_weights
        from accelerate.big_modeling import init_empty_weights
        init_contexts.append(no_init_weights(_enable=load))
        init_contexts.append(init_empty_weights())

    with ContextManagers(init_contexts):
        for idx, weight in enumerate(weight_split):
            new_linear = torch.nn.Linear(weight.size(1),
                                         weight.size(0),
                                         bias=False if bias is None else True)
            new_linear.bias = bias
            new_linear.weight = torch.nn.Parameter(weight.contiguous(), requires_grad=False)
            linear_list.add_module(f"{module_name}_dq_{idx}", new_linear)
    return linear_list


def split_linears(module: torch.nn.Module, n_splits_hidden_size=2, n_splits_down_proj=2,
                  load=False):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Attention
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
    attn_module_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    baichuan_attn_module_names = ["W_pack", "o_proj"]
    mlp_module_names = ["down_proj", "up_proj", "gate_proj"]
    if (
        isinstance(module, (Qwen2Attention, LlamaAttention))
        or module.__class__.__name__ in ['MiniCPMAttention']
    ):
        for name in attn_module_names:
            setattr(module, f"{name}_dq_list", split_linear(getattr(module, name), name,
                                                            n_splits=n_splits_hidden_size,
                                                            load=load))
            delattr(module, name)
    elif (
        isinstance(module, (Qwen2MLP, LlamaMLP))
        or module.__class__.__name__ in ['MiniCPMMLP', 'MLP']
    ):
        for name in mlp_module_names:
            n_splits_mlp = n_splits_hidden_size
            if name == 'down_proj':
                n_splits_mlp = n_splits_down_proj
            setattr(module, f"{name}_dq_list", split_linear(getattr(module, name), name,
                                                            n_splits=n_splits_mlp,
                                                            load=load))
            delattr(module, name)
    elif module.__class__.__name__ == 'Attention' and module.config.model_type == 'baichuan':
        # baichuan attention
        for name in baichuan_attn_module_names:
            setattr(module, f"{name}_dq_list", split_linear(getattr(module, name), name,
                                                            n_splits=n_splits_hidden_size,
                                                            load=load))
            delattr(module, name)

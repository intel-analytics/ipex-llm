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
from intel_npu_acceleration_library.nn import QuantizedLinear


def module_optimization(func) -> torch.nn.Module:
    """Optimize recursively a torch.nn.Module with a specific function.

    The function `func` get called recursively to every module in the network.

    Args:
        func (Callable): optimization function

    Returns:
        torch.nn.Module: optimized module
    """

    def wrapper(model: torch.nn.Module, qtype, *args, **kwargs):
        """Recursively apply the optimization function.

        Args:
            model (torch.nn.Module): original module
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        """
        for name, layer in model.named_children():
            new_layer = func(layer, qtype, *args, **kwargs)
            if new_layer:
                model.add_module(name, new_layer)
                wrapper(new_layer, qtype, *args, **kwargs)
            else:
                wrapper(layer, qtype, *args, **kwargs)

    return wrapper


@module_optimization
def replace_with_QuantizedLinear(layer, qtype):
    from ipex_llm.transformers.low_bit_linear import ggml_convert_qtype
    from ipex_llm.ggml.quantize import ggml_tensor_qtype
    iqtype = ggml_tensor_qtype[qtype]
    if isinstance(layer, torch.nn.Linear):
        qweights, scale = ggml_convert_qtype(layer.weight.data, iqtype, 'cpu')
        return QuantizedLinear(qweights, scale, layer.bias)


def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)


def optimize_llm(model: torch.nn.Module):
    if model.config.model_type == "llama":
        from ipex_llm.transformers.npu_models.llama import merge_qkv
        model.apply(merge_qkv)
        from ipex_llm.transformers.npu_models.llama import merge_mlp
        model.apply(merge_mlp)
        from ipex_llm.transformers.npu_models.llama import llama_model_forward
        from transformers.models.llama.modeling_llama import LlamaModel
        convert_forward(model, LlamaModel, llama_model_forward)
        from ipex_llm.transformers.npu_models.llama import llama_attention_forward
        from transformers.models.llama.modeling_llama import LlamaAttention
        convert_forward(model, LlamaAttention, llama_attention_forward)
        from ipex_llm.transformers.npu_models.llama import llama_mlp_forward
        from transformers.models.llama.modeling_llama import LlamaMLP
        convert_forward(model, LlamaMLP, llama_mlp_forward)

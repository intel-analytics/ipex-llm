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
import importlib
from ipex_llm.transformers.npu_models.linear import QuantizedLinear


def module_optimization(func) -> torch.nn.Module:
    """Optimize recursively a torch.nn.Module with a specific function.

    The function `func` get called recursively to every module in the network.

    Args:
        func (Callable): optimization function

    Returns:
        torch.nn.Module: optimized module
    """

    def wrapper(model: torch.nn.Module, qtype, device, modules_to_not_convert, *args, **kwargs):
        """Recursively apply the optimization function.

        Args:
            model (torch.nn.Module): original module
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        """
        for name, layer in model.named_children():
            if name not in modules_to_not_convert:
                new_layer = func(layer, qtype, device, modules_to_not_convert, *args, **kwargs)
                if new_layer:
                    model.add_module(name, new_layer)
                    wrapper(new_layer, qtype, device, modules_to_not_convert, *args, **kwargs)
                else:
                    wrapper(layer, qtype, device, modules_to_not_convert, *args, **kwargs)

    return wrapper


@module_optimization
def replace_with_QuantizedLinear(layer, qtype, device, modules_to_not_convert):
    from ipex_llm.transformers.low_bit_linear import ggml_convert_qtype
    from ipex_llm.ggml.quantize import ggml_tensor_qtype
    iqtype = ggml_tensor_qtype[qtype]
    if isinstance(layer, torch.nn.Linear) and not hasattr(layer, "qtype"):
        if qtype == "sym_int4_rtn":
            # workaround for qwen2 & int4
            if (layer.in_features == 3584 and layer.out_features == 152064) or \
               (layer.in_features == 18944 and layer.out_features == 3584):
                qtype = "sym_int8_rtn"
                iqtype = ggml_tensor_qtype[qtype]
        qweights, scale = ggml_convert_qtype(layer.weight.data.to(torch.float32),
                                             iqtype, device=device)
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
        from ipex_llm.transformers.npu_models.llama import merge_mlp
        from ipex_llm.transformers.npu_models.llama import llama_model_forward
        from ipex_llm.transformers.npu_models.llama import llama_attention_forward
        from ipex_llm.transformers.npu_models.llama import llama_mlp_forward
        from transformers.models.llama.modeling_llama import LlamaModel
        from transformers.models.llama.modeling_llama import LlamaAttention
        from transformers.models.llama.modeling_llama import LlamaMLP
        model.apply(merge_qkv)
        model.apply(merge_mlp)
        convert_forward(model, LlamaModel, llama_model_forward)
        convert_forward(model, LlamaAttention, llama_attention_forward)
        convert_forward(model, LlamaMLP, llama_mlp_forward)

    elif model.config.model_type == "mistral":
        from ipex_llm.transformers.npu_models.mistral import merge_qkv
        from ipex_llm.transformers.npu_models.mistral import merge_mlp
        model.apply(merge_qkv)
        model.apply(merge_mlp)

        from ipex_llm.transformers.npu_models.mistral import mistral_model_forward
        from ipex_llm.transformers.npu_models.mistral import mistral_attention_forward
        from ipex_llm.transformers.npu_models.mistral import mistral_mlp_forward
        from transformers.models.mistral.modeling_mistral import MistralModel
        from transformers.models.mistral.modeling_mistral import MistralAttention
        from transformers.models.mistral.modeling_mistral import MistralMLP
        convert_forward(model, MistralModel, mistral_model_forward)
        convert_forward(model, MistralAttention, mistral_attention_forward)
        convert_forward(model, MistralMLP, mistral_mlp_forward)

    elif model.config.model_type == "qwen2":
        from ipex_llm.transformers.npu_models.qwen2 import merge_qkv
        from ipex_llm.transformers.npu_models.qwen2 import merge_mlp
        model.apply(merge_qkv)
        model.apply(merge_mlp)

        from ipex_llm.transformers.npu_models.qwen2 import qwen2_model_forward
        from ipex_llm.transformers.npu_models.qwen2 import qwen2_attention_forward
        from ipex_llm.transformers.npu_models.qwen2 import qwen2_mlp_forward
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2SdpaAttention
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
        convert_forward(model, Qwen2Model, qwen2_model_forward)
        convert_forward(model, Qwen2Attention, qwen2_attention_forward)
        convert_forward(model, Qwen2SdpaAttention, qwen2_attention_forward)
        convert_forward(model, Qwen2MLP, qwen2_mlp_forward)

    elif model.config.model_type == "minicpm":
        from ipex_llm.transformers.npu_models.minicpm import merge_qkv
        from ipex_llm.transformers.npu_models.minicpm import merge_mlp
        from ipex_llm.transformers.npu_models.minicpm import padding_lm_head
        model.apply(merge_qkv)
        model.apply(merge_mlp)
        model.apply(padding_lm_head)

        from ipex_llm.transformers.npu_models.minicpm import minicpm_model_causal_lm_forward
        from ipex_llm.transformers.npu_models.minicpm import minicpm_attention_forward
        from ipex_llm.transformers.npu_models.minicpm import minicpm_mlp_forward
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        convert_forward(model, module.MiniCPMForCausalLM, minicpm_model_causal_lm_forward)
        convert_forward(model, module.MiniCPMAttention, minicpm_attention_forward)
        convert_forward(model, module.MiniCPMMLP, minicpm_mlp_forward)

    elif model.config.model_type == "chatglm":
        if model.config.num_layers == 40 and hasattr(model.config, 'rope_ratio'):
            # glm-4-9b
            from ipex_llm.transformers.npu_models.chatglm4 import chatglm4_model_forward
            from ipex_llm.transformers.npu_models.chatglm4 import chatglm4_attention_forward
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            convert_forward(model, module.ChatGLMModel, chatglm4_model_forward)
            convert_forward(model, module.SelfAttention, chatglm4_attention_forward)
        else:
            # chatglm-3-6b
            from ipex_llm.transformers.npu_models.chatglm import chatglm2_model_forward
            from ipex_llm.transformers.npu_models.chatglm import chatglm2_attention_forward
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            convert_forward(model, module.ChatGLMModel, chatglm2_model_forward)
            convert_forward(model, module.SelfAttention, chatglm2_attention_forward)

    elif model.config.model_type == "stablelm":
        from ipex_llm.transformers.npu_models.stablelm import merge_qkv
        from ipex_llm.transformers.npu_models.stablelm import merge_mlp
        model.apply(merge_qkv)
        model.apply(merge_mlp)

        from ipex_llm.transformers.npu_models.stablelm import stablelm_model_forward
        from ipex_llm.transformers.npu_models.stablelm import stablelm_attention_forward
        from ipex_llm.transformers.npu_models.stablelm import stablelm_mlp_forward
        from transformers.models.stablelm.modeling_stablelm import StableLmModel
        from transformers.models.stablelm.modeling_stablelm import StableLmAttention
        from transformers.models.stablelm.modeling_stablelm import StableLmMLP
        convert_forward(model, StableLmModel, stablelm_model_forward)
        convert_forward(model, StableLmAttention, stablelm_attention_forward)
        convert_forward(model, StableLmMLP, stablelm_mlp_forward)

    elif model.config.model_type == "baichuan":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.npu_models.baichuan import baichuan_mlp_forward, merge_mlp
        from ipex_llm.transformers.npu_models.baichuan import baichuan_attention_fwd
        model.apply(merge_mlp)

        convert_forward(model, module.MLP, baichuan_mlp_forward)
        convert_forward(model, module.Attention, baichuan_attention_fwd)

    elif model.config.model_type == "phi3_v":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.npu_models.phi3_v import merge_qkv
        from ipex_llm.transformers.npu_models.phi3_v import phi3v_encoder_attention_forward
        from ipex_llm.transformers.npu_models.phi3_v import phi3v_model_forward
        model.apply(merge_qkv)

        from transformers.models.clip.modeling_clip import CLIPAttention
        convert_forward(model, CLIPAttention, phi3v_encoder_attention_forward)
        convert_forward(model, module.Phi3VModel, phi3v_model_forward)

        from ipex_llm.transformers.npu_models.phi3 import phi3_attention_forward
        convert_forward(model, module.Phi3Attention, phi3_attention_forward)

    elif model.config.model_type == "phi3":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.npu_models.phi3 import phi3_attention_forward

        convert_forward(model, module.Phi3Attention, phi3_attention_forward)


def optimize_llm_post(model: torch.nn.Module):
    # experimental support for fused decoderlayer implementation
    if model.config.model_type == "llama":
        model.model.embed_tokens.to(torch.float32)
        model.model.norm.to(torch.float32)
        model.lm_head.to(torch.float32)

        from ipex_llm.transformers.low_bit_linear import LowBitLinear, \
            ggml_tensor_qtype, FP4Params

        if isinstance(model.lm_head, torch.nn.Linear):
            new_linear = LowBitLinear(model.lm_head.in_features,
                                      model.lm_head.out_features,
                                      ggml_tensor_qtype["sym_int4"],
                                      False)
            paramsLowBit = FP4Params(data=model.lm_head.weight.data,
                                     requires_grad=False,
                                     quantized=False,
                                     _shape=None,
                                     qtype=ggml_tensor_qtype["sym_int4"],
                                     in_features=model.lm_head.in_features).to("cpu")
            new_linear._parameters['weight'] = paramsLowBit
            model.lm_head = new_linear

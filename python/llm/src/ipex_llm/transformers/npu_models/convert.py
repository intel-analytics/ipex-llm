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


from ipex_llm.utils.common.log4Error import invalidInputError
import os
import torch
import importlib
from ipex_llm.transformers.npu_models.linear import QuantizedLinear
import tempfile
import time
from typing import Callable, List, Optional
from transformers import GenerationConfig, \
    LogitsProcessorList, StoppingCriteriaList


def module_optimization(func) -> torch.nn.Module:
    """Optimize recursively a torch.nn.Module with a specific function.

    The function `func` get called recursively to every module in the network.

    Args:
        func (Callable): optimization function

    Returns:
        torch.nn.Module: optimized module
    """

    def wrapper(model: torch.nn.Module, qtype, device, modules_to_not_convert,
                group_size=0, *args, **kwargs):
        """Recursively apply the optimization function.

        Args:
            model (torch.nn.Module): original module
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        """
        for name, layer in model.named_children():
            if name not in modules_to_not_convert:
                new_layer = func(layer, qtype, device, modules_to_not_convert,
                                 group_size=group_size, *args, **kwargs)
                if new_layer:
                    model.add_module(name, new_layer)
                    wrapper(new_layer, qtype, device, modules_to_not_convert,
                            group_size=group_size, *args, **kwargs)
                else:
                    wrapper(layer, qtype, device, modules_to_not_convert,
                            group_size=group_size, *args, **kwargs)

    return wrapper


@module_optimization
def replace_with_QuantizedLinear(layer, qtype, device, modules_to_not_convert,
                                 group_size):
    from ipex_llm.transformers.low_bit_linear import ggml_convert_qtype
    from ipex_llm.ggml.quantize import ggml_tensor_qtype
    iqtype = ggml_tensor_qtype[qtype]
    if isinstance(layer, torch.nn.Linear) and not hasattr(layer, "qtype"):
        if qtype == "sym_int4_rtn":
            # workaround for qwen2-7B & int4
            if (layer.in_features == 3584 and layer.out_features == 152064) or \
               (layer.in_features == 18944 and layer.out_features == 3584):
                qtype = "sym_int8_rtn"
                iqtype = ggml_tensor_qtype[qtype]
        enable_scale_search = os.environ.get("IPEX_LLM_NPU_QUANTIZATION_OPT", "0") != "0"
        qweights, scale = ggml_convert_qtype(layer.weight.data.to(torch.float32),
                                             iqtype, device=device,
                                             enable_scale_search=enable_scale_search)
        return QuantizedLinear(qweights, scale, layer.bias,
                               group_size=group_size)


@module_optimization
def replace_with_DequantizedLinear(layer, qtype, device, modules_to_not_convert,
                                   group_size):
    from ipex_llm.transformers.npu_models.linear import DequantizedLinear
    from ipex_llm.transformers.low_bit_linear import ggml_convert_qtype
    from ipex_llm.ggml.quantize import ggml_tensor_qtype
    iqtype = ggml_tensor_qtype[qtype]
    if isinstance(layer, torch.nn.Linear) and not hasattr(layer, "qtype"):
        enable_scale_search = os.environ.get("IPEX_LLM_NPU_QUANTIZATION_OPT", "0") != "0"
        qweights, scale = ggml_convert_qtype(layer.weight.data.to(torch.float32),
                                             iqtype, device=device,
                                             enable_scale_search=enable_scale_search)
        return DequantizedLinear(qweights, scale, layer.bias)


@module_optimization
def replace_with_FP16Linear(layer, qtype, device, modules_to_not_convert,
                            group_size):
    from ipex_llm.transformers.npu_models.linear import Linear
    if isinstance(layer, torch.nn.Linear) and not hasattr(layer, "qtype"):
        return Linear(layer.weight, layer.bias)


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


def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]=None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
):
    # if do_print=True, output timing message
    do_print = kwargs.pop("do_print", False)
    time_t1 = time.perf_counter()
    new_generate_kwargs = {}
    for var in ['max_new_tokens', 'attention_mask', 'eos_token_id']:
        value = kwargs.pop(var, None)
        if value is not None:
            new_generate_kwargs[var] = value

    if isinstance(inputs[0], torch.Tensor):
        input_list = inputs[0].flatten().tolist()
    else:
        input_list = inputs[0]
    input_length = len(input_list)

    new_tokens = new_generate_kwargs['max_new_tokens']
    invalidInputError(input_length + new_tokens <= self.kv_len + 1,
                      "Input plus output tokens should not exceed max_context_len.")
    # TODO: may optimize this part later
    invalidInputError(new_tokens < 1024,
                      f"Generated tokens ({new_tokens}) exceed named pipeline limitation.")

    if "eos_token_id" not in new_generate_kwargs:
        eos = 0xffffffff
    else:
        eos = new_generate_kwargs["eos_token_id"]
    output_tokens = []
    from .npu_llm_cpp import run_decode, run_prefill, reset

    token = run_prefill(self.model_ptr, input_list, self.vocab_size)
    idx = 1
    time_t2 = time.perf_counter()
    output_tokens.append(torch.tensor([token]))
    for i in range(new_tokens - 1):
        if token == eos:
            break
        token = run_decode(self.model_ptr, token, self.vocab_size)
        idx += 1
        output_tokens.append(torch.tensor([token]))
    output = torch.stack(output_tokens, dim=1)
    output = torch.cat((inputs, output), dim=1)
    time_t3 = time.perf_counter()

    reset(self.model_ptr)
    self.first_cost = time_t2 - time_t1  # seconds
    self.rest_cost_mean = (time_t3 - time_t2) / (idx - 1)  # seconds
    self.encoder_time = 0.0

    if do_print:
        print(f" Number of input tokens: {input_length}")
        print(f" Generated tokens: {idx}")
        print(f" First token generation time: {(time_t2 - time_t1):.2f} s")
        print(f" Generation average latency: {(time_t3 - time_t2) * 1000 /(idx - 1):.2f} ms, "
              f"({(idx - 1)/(time_t3 - time_t2):.2f} token/s)")
        print(f" Generation time: {(time_t3 - time_t1):.2f} s\n")

    return output


def optimize_llm_single_process(
    model: torch.nn.Module,
    kv_len: int,
    max_prompt_len: int,
    transpose_value_cache: bool,
    group_size: int,
    qtype: str,
    save_directory: str,
    fuse_layers: int=None
):
    from ipex_llm.transformers.npu_pipeline_model.convert_pipeline import convert_llm
    from .npu_llm_cpp import load_model_from_file

    convert_llm(model,
                kv_len=kv_len,
                max_prompt_len=max_prompt_len,
                transpose_value_cache=transpose_value_cache,
                group_size=group_size,
                qtype=qtype,
                convert_model=True,
                save_directory=save_directory,
                fuse_layers=fuse_layers)
    try:
        model_ptr = load_model_from_file(save_directory)
        model.kv_len = kv_len
        model.model_ptr = model_ptr
        model.vocab_size = model.config.vocab_size
    except:
        invalidInputError(False,
                          "False to InitLLMPipeline.")
    # patch generate function
    import types
    model.generate = types.MethodType(generate, model)
    return model

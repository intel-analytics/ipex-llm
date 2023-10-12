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
# https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/utils/bitsandbytes.py
# and https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
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


import torch
import torch.nn as nn
from accelerate import init_empty_weights
import warnings
import transformers
import importlib
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from .utils import logger


def _replace_with_low_bit_linear(model, qtype, modules_to_not_convert=None,
                                 current_key_name=None, convert_shape_only=False):
    from bigdl.llm.transformers.low_bit_linear import LowBitLinear, FP4Params, FP16Linear
    has_been_replaced = False

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    new_linear = None
                    if qtype != ggml_tensor_qtype["fp16"]:
                        new_linear = LowBitLinear(
                            module.in_features,
                            module.out_features,
                            qtype,
                            module.bias is not None,
                        )

                        device_type = module.weight.data.device.type
                        # Copy the weights
                        paramsLowBit = FP4Params(data=module.weight.data,
                                                 requires_grad=False,
                                                 quantized=False,
                                                 _shape=None,
                                                 convert_shape_only=convert_shape_only,
                                                 qtype=qtype).to(device_type)
                        new_linear._parameters['weight'] = paramsLowBit
                    else:
                        #  only support two size now
                        #  may generalize to other sizes
                        if module.in_features in [4096, 11008]:
                            # esimd fp16 path
                            new_linear = FP16Linear(
                                module.in_features,
                                module.out_features,
                                qtype,
                                module.bias is not None,
                            )
                            device_type = module.weight.data.device.type

                            # convert here
                            m, n = module.weight.data.shape
                            trans_weight = module.weight.data.reshape(m//16, 16, n)
                            trans_weight = trans_weight.transpose(1, 2).contiguous()
                            new_linear._parameters['weight'] = nn.Parameter(trans_weight)

                    #  fp16 may generalize to other sizes later
                    if new_linear is not None:
                        if module.bias is not None:
                            new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                                .to(device_type)

                        model._modules[name] = new_linear
                        has_been_replaced = True
                        # Force requires grad to False to avoid unexpected errors
                        model._modules[name].requires_grad_(False)

                        module.weight = None

        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            _, _flag = _replace_with_low_bit_linear(
                module,
                qtype,
                modules_to_not_convert,
                current_key_name,
                convert_shape_only,
            )
            has_been_replaced = _flag or has_been_replaced
    return model, has_been_replaced


def ggml_convert_low_bit(model, qtype, optimize_model=True,
                         convert_shape_only=False, device="cpu",
                         modules_to_not_convert=None):
    modules_to_not_convert = [] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_low_bit_linear(
        model, qtype, modules_to_not_convert,
        None, convert_shape_only,
    )
    if not has_been_replaced:
        warnings.warn(
            "No linear modules were found in "
            "your model. This can happen for some architectures such as gpt2 that uses Conv1D "
            "instead of Linear layers. Please double check your model architecture, or submit "
            "an issue on github if you think this is a bug."
        )
    elif device == "cpu":
        model.to(torch.float32)
    elif device == "meta":
        # Do nothing here for weights are empty.
        pass

    if optimize_model:
        model = optimize(model)
    return model


def convert_forward(m, target_m, new_forward):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            bound_method = new_forward.__get__(sub_m, sub_m.__class__)
            setattr(sub_m, "forward", bound_method)
        convert_forward(sub_m, target_m, new_forward)


def optimize(model):
    from packaging import version
    from bigdl.llm.transformers.models.llama import llama_attention_forward_4_31
    from bigdl.llm.transformers.models.llama import llama_rms_norm_forward
    from transformers.modeling_utils import PreTrainedModel

    # All huggingface format models are inherited from `PreTrainedModel`
    if not isinstance(model, PreTrainedModel):
        logger.info("Only HuggingFace Transformers models are currently "
                    "supported for further optimizations")
        return model

    trans_version = transformers.__version__
    if version.parse(trans_version) >= version.parse("4.31.0"):
        convert_forward(
            model,
            transformers.models.llama.modeling_llama.LlamaAttention,
            llama_attention_forward_4_31,)
        convert_forward(
            model,
            transformers.models.llama.modeling_llama.LlamaRMSNorm,
            llama_rms_norm_forward,)
    else:
        # todo implement 4.28.0 ~ 4.30.2
        pass

    if model.config.architectures[0] == "ChatGLMModel":
        if hasattr(model.config, "padded_vocab_size") and model.config.padded_vocab_size == 65024:
            # chatglm2-6b
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.chatglm2 import chatglm2_attention_forward_8eb45c
            from bigdl.llm.transformers.models.chatglm2 import core_attn_forward_8eb45c
            convert_forward(model,
                            module.SelfAttention,
                            chatglm2_attention_forward_8eb45c
                            )
            convert_forward(model,
                            module.CoreAttention,
                            core_attn_forward_8eb45c)
        elif model.config.vocab_size == 130528:
            # chatglm-6b
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.chatglm import chatglm_attention_forward
            convert_forward(model,
                            module.SelfAttention,
                            chatglm_attention_forward
                            )
    elif "mpt" in model.config.model_type:
        modeling_module_name = model.__class__.__module__
        attention_module_name = '.'.join(modeling_module_name.split('.')[:-1]) + ".attention"
        module = importlib.import_module(attention_module_name)
        from bigdl.llm.transformers.models.mpt import mpt_multihead_attention_forward
        convert_forward(model,
                        module.MultiheadAttention,
                        mpt_multihead_attention_forward
                        )
    elif "gptj" in model.config.model_type:
        # dolly-v1-6b
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.gptj import gptj_attention_forward
        convert_forward(model,
                        module.GPTJAttention,
                        gptj_attention_forward)
    elif "bloom" in model.config.model_type:
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.bloom import bloom_attention_forward
        convert_forward(model,
                        module.BloomAttention,
                        bloom_attention_forward
                        )
    elif "falcon" in model.config.model_type or "RefinedWeb" in model.config.model_type:
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        if "RWForCausalLM" in model.config.architectures:
            if hasattr(model.config, "multi_query"):
                # falcon-7b need to check performance drop after kv cache support.
                # from bigdl.llm.transformers.models.falcon import rw_attention_forward_7b
                # convert_forward(model,
                #                 module.Attention,
                #                 rw_attention_forward_7b
                #                 )
                pass
            else:
                # falcon-40b
                from bigdl.llm.transformers.models.falcon import rw_attention_forward_40b
                convert_forward(model,
                                module.Attention,
                                rw_attention_forward_40b
                                )
        elif "FalconForCausalLM" in model.config.architectures:
            # falcon-180b
            from bigdl.llm.transformers.models.falcon import falcon_attention_forward
            convert_forward(model,
                            module.FalconAttention,
                            falcon_attention_forward
                            )
    elif model.config.model_type == "baichuan" and model.config.vocab_size == 125696:
        # baichuan2
        if model.config.hidden_size == 4096:
            # baichuan2-7B
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.baichuan2 import baichuan_attention_forward_7b
            convert_forward(model,
                            module.Attention,
                            baichuan_attention_forward_7b
                            )
        elif model.config.hidden_size == 5120:
            # baichuan2-13B
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.baichuan2 import baichuan_attention_forward_13b
            convert_forward(model,
                            module.BaichuanAttention,
                            baichuan_attention_forward_13b
                            )

    elif model.config.model_type == "baichuan":
        # baichuan1
        if model.config.hidden_size == 4096:
            # baichuan-7B
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.baichuan import baichuan_attention_forward_7b
            convert_forward(model,
                            module.Attention,
                            baichuan_attention_forward_7b
                            )
        elif model.config.hidden_size == 5120:
            # baichuan-13B
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.baichuan import baichuan_attention_forward_13b
            convert_forward(model,
                            module.BaichuanAttention,
                            baichuan_attention_forward_13b
                            )

    elif model.config.model_type == "gpt_neox":
        from bigdl.llm.transformers.models.gptneox import gptneox_attention_forward
        convert_forward(model,
                        transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention,
                        gptneox_attention_forward
                        )
    elif model.config.model_type == "internlm":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.internlm import internlm_attention_forward
        convert_forward(model,
                        module.InternLMAttention,
                        internlm_attention_forward
                        )
    elif model.config.model_type == "qwen":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.qwen import qwen_attention_forward
        convert_forward(model,
                        module.QWenAttention,
                        qwen_attention_forward
                        )
    elif model.config.model_type == "aquila":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.aquila import aquila_attention_forward
        convert_forward(model,
                        module.AquilaAttention,
                        aquila_attention_forward
                        )
    return model

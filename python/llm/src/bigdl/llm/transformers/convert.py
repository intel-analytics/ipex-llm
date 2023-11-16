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


import platform
import torch
import torch.nn as nn
from accelerate import init_empty_weights
import warnings
import transformers
import importlib.util
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from .utils import logger
from typing import Union
import numpy as np
from bigdl.llm.utils.common import invalidInputError


def is_auto_gptq_available():
    return importlib.util.find_spec("auto_gptq") is not None


def is_auto_awq_available():
    return importlib.util.find_spec("awq") is not None


def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


if is_auto_gptq_available():
    from auto_gptq.utils.peft_utils import QuantLinearCuda, QuantLinearCudaOld


if is_auto_awq_available():
    from bigdl.llm.transformers.awq.linear import WQLinear_GEMM


def is_linear_module(module):

    in_features = None
    out_features = None
    mp_group = None

    is_awq = is_auto_awq_available() and isinstance(module, WQLinear_GEMM)

    if is_auto_gptq_available() and isinstance(module, QuantLinearCudaOld):
        in_features = module.infeatures
        out_features = module.outfeatures
        mp_group = None
        result = True
    elif isinstance(module, nn.Linear) or is_awq:
        in_features = module.in_features
        out_features = module.out_features
        mp_group = None
        result = True
    else:
        if is_deepspeed_available():
            from deepspeed.module_inject.layers import LinearLayer, LinearAllreduce
            if isinstance(module, LinearLayer):
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]
                mp_group = None
                result = True
            elif isinstance(module, LinearAllreduce):
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]
                mp_group = module.mp_group
                result = True
            else:
                result = False
        else:
            result = False

    return result, (in_features, out_features, mp_group)


from bigdl.llm.transformers.low_bit_linear import get_ggml_qk_size
Q4_1 = get_ggml_qk_size("asym_int4")


def convert_gptq(module, awq=False):
    scales = module.scales

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(module.qzeros, 2).expand(-1, -1, 32 // module.bits),
        module.wf.unsqueeze(0)).to(torch.int16 if module.bits == 8 else torch.int8)
    zeros = torch.bitwise_and(zeros, (2 ** module.bits) - 1)

    if not awq:
        zeros = zeros + 1
    zeros = zeros.reshape(scales.shape)

    if awq:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(module.qweight, 2).expand(-1, -1, 32 // module.bits),
            module.wf.unsqueeze(0)).to(torch.int16 if module.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2 ** module.bits) - 1)
        weight = weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])
    else:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(module.qweight, 1).expand(-1, 32 // module.bits, -1),
            module.wf.unsqueeze(-1)).to(torch.int8)
        weight = torch.bitwise_and(weight, (2 ** module.bits) - 1)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    # convert weight to ggml format
    weight = weight.reshape(weight.shape[0]//module.group_size, module.group_size, weight.shape[1])
    weight = weight.permute(2, 0, 1).reshape(weight.shape[2], -1, 2, Q4_1//2)
    weight = weight.transpose(2, 3)
    weight = torch.bitwise_left_shift(weight,
                                      torch.tensor([0, 4], dtype=torch.int8).reshape(1, 1, 1, 2))
    weight = torch.bitwise_or(weight[:, :, :, 0], weight[:, :, :, 1]).contiguous()

    # convert zeros to ggml format
    zeros = zeros.reshape(-1, 1, zeros.shape[1]).permute(2, 0, 1)\
        .unsqueeze(2)\
        .expand(-1, -1, module.group_size//Q4_1, -1)\
        .reshape(zeros.shape[1], -1, 1)\
        .contiguous().to(torch.float16)

    # convert scales to ggml format
    scales = scales.reshape(-1, 1, scales.shape[1]).permute(2, 0, 1)\
        .unsqueeze(2)\
        .expand(-1, -1, module.group_size//Q4_1, -1)\
        .reshape(scales.shape[-1], -1, 1)\
        .contiguous().to(torch.float16)

    m = -(zeros * scales)
    d = scales

    ggml_weight = torch.cat([d.view(torch.uint8),
                             m.view(torch.uint8),
                             weight.view(torch.uint8)], dim=-1)
    ggml_weight = ggml_weight.reshape([-1])

    return ggml_weight


def _replace_with_low_bit_linear(model, qtype, modules_to_not_convert=None,
                                 current_key_name=None, convert_shape_only=False,
                                 replace_embedding=False):
    from bigdl.llm.transformers.low_bit_linear import LowBitLinear, FP4Params, FP16Linear
    from bigdl.llm.transformers.embedding import LLMEmbedding
    has_been_replaced = False

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        is_linear, linear_args = is_linear_module(module)
        if is_linear and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                in_features, out_features, mp_group = linear_args
                with init_empty_weights():
                    new_linear = None
                    is_gptq = is_auto_gptq_available() and isinstance(module, QuantLinearCudaOld)
                    is_awq = is_auto_awq_available() and isinstance(module, WQLinear_GEMM)
                    if is_gptq or is_awq:
                        has_bias = module.bias is not None and module.bias.abs().sum() != 0
                        new_linear = LowBitLinear(
                            in_features,
                            out_features,
                            qtype=qtype,
                            bias=has_bias,
                            mp_group=mp_group,
                        )
                        device_type = module.qweight.data.device.type
                        invalidInputError(device_type != "meta",
                                          "converting from meta device is not supported")
                        # Copy the weights
                        paramsLowBit = FP4Params(data=convert_gptq(module, awq=is_awq),
                                                 requires_grad=False,
                                                 quantized=True,
                                                 _shape=(out_features, in_features),
                                                 convert_shape_only=convert_shape_only,
                                                 qtype=qtype).to(device_type)
                        new_linear._parameters['weight'] = paramsLowBit
                        if has_bias:
                            new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                                .to(device_type)
                    elif qtype != ggml_tensor_qtype["fp16"]:
                        new_linear = LowBitLinear(
                            in_features,
                            out_features,
                            qtype,
                            module.bias is not None,
                            mp_group=mp_group,
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
                        if module.bias is not None:
                            new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                                .to(device_type)
                    else:
                        #  only support two size now
                        #  may generalize to other sizes
                        if module.in_features in [4096, 11008]:
                            # esimd fp16 path
                            new_linear = FP16Linear(
                                in_features,
                                out_features,
                                qtype,
                                module.bias is not None,
                                mp_group=mp_group,
                            )
                            device_type = module.weight.data.device.type

                            # convert here
                            m, n = module.weight.data.shape
                            trans_weight = module.weight.data.reshape(m//16, 16, n)
                            trans_weight = trans_weight.transpose(1, 2).contiguous()
                            new_linear._parameters['weight'] = nn.Parameter(trans_weight)
                            if module.bias is not None:
                                new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                                    .to(device_type)

                    #  fp16 may generalize to other sizes later
                    if new_linear is not None:
                        model._modules[name] = new_linear
                        has_been_replaced = True
                        # Force requires grad to False to avoid unexpected errors
                        model._modules[name].requires_grad_(False)

                        module.weight = None
        elif replace_embedding and type(module) == nn.Embedding:
            # skip user-defined Embedding layer
            if platform.system().lower() == 'windows':
                model._modules[name] = LLMEmbedding(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    padding_idx=module.padding_idx,
                    max_norm=module.max_norm,
                    norm_type=module.norm_type,
                    scale_grad_by_freq=module.scale_grad_by_freq,
                    sparse=module.sparse,
                    _weight=module.weight.data,
                )

        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            _, _flag = _replace_with_low_bit_linear(
                module,
                qtype,
                modules_to_not_convert,
                current_key_name,
                convert_shape_only,
                replace_embedding,
            )
            has_been_replaced = _flag or has_been_replaced
    return model, has_been_replaced


def _optimize_pre(model):
    from transformers.modeling_utils import PreTrainedModel
    # All huggingface format models are inherited from `PreTrainedModel`
    if not isinstance(model, PreTrainedModel):
        logger.info("Only HuggingFace Transformers models are currently "
                    "supported for further optimizations")
        return model
    # process NormHead module in Baichuan2 7B and 13B
    if model.config.model_type == "baichuan" and model.config.vocab_size == 125696:
        # NormHead do normalization on the weights just once at inference time.
        # so we do it in advance and convert it to Linear so that it can be replaced.
        # modeling_module_name = model.__class__.__module__
        # module = importlib.import_module(modeling_module_name)
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            # do we need to check the class instance?
            vocab_size, hidden_size = model.lm_head.weight.shape
            lm_head_weight_data = model.lm_head.weight.data
            model.lm_head = nn.Linear(hidden_size, vocab_size, bias=False,
                                      device=lm_head_weight_data.device)
            # In which case we are NOT loading the normalized weights
            if model.lm_head.weight.data.device != "meta":
                norm_weight = nn.functional.normalize(lm_head_weight_data)
                model.lm_head.weight.data = norm_weight
    return model


def ggml_convert_low_bit(model, qtype, optimize_model=True,
                         convert_shape_only=False, device="cpu",
                         modules_to_not_convert=None, replace_embedding=False):
    logger.info(f"Converting the current model to "
                f"{list(ggml_tensor_qtype.keys())[list(ggml_tensor_qtype.values()).index(qtype)]} "
                f"format......")
    modules_to_not_convert = [] if modules_to_not_convert is None else modules_to_not_convert

    if optimize_model:
        model = _optimize_pre(model)

    model, has_been_replaced = _replace_with_low_bit_linear(
        model, qtype, modules_to_not_convert,
        None, convert_shape_only, replace_embedding,
    )
    if not has_been_replaced:
        warnings.warn(
            "No linear modules were found in "
            "your model. This can happen for some architectures such as gpt2 that uses Conv1D "
            "instead of Linear layers. Please double check your model architecture, or submit "
            "an issue on github if you think this is a bug."
        )
    elif device == "cpu":
        if not (getattr(model, "quantization_method", None) == "gptq"):
            model.to(torch.float32)
    elif device == "meta":
        # Do nothing here for weights are empty.
        pass

    if optimize_model:
        model = _optimize_post(model)
    return model


def convert_forward(m, target_m, new_forward):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            bound_method = new_forward.__get__(sub_m, sub_m.__class__)
            setattr(sub_m, "forward", bound_method)
        convert_forward(sub_m, target_m, new_forward)


def _optimize_post(model):
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

    if model.config.architectures is not None and model.config.architectures[0] == "ChatGLMModel":
        if model.config.num_layers == 28 and hasattr(model.config, 'rope_ratio'):
            # chatglm2-6b-32k
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.chatglm2_32k import chatglm2_32k_attention_forward
            convert_forward(model,
                            module.SelfAttention,
                            chatglm2_32k_attention_forward)
        elif hasattr(model.config, 'padded_vocab_size') and \
                model.config.padded_vocab_size == 65024:
            # chatglm2-6b
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.chatglm2 import chatglm2_attention_forward_8eb45c
            from bigdl.llm.transformers.models.chatglm2 import core_attn_forward_8eb45c
            from bigdl.llm.transformers.models.chatglm2 import chatglm_rms_norm_forward
            from bigdl.llm.transformers.models.chatglm2 import chatglm2_model_forward
            convert_forward(model,
                            module.SelfAttention,
                            chatglm2_attention_forward_8eb45c
                            )
            convert_forward(model,
                            module.CoreAttention,
                            core_attn_forward_8eb45c)
            convert_forward(model,
                            module.ChatGLMModel,
                            chatglm2_model_forward)
            convert_forward(model,
                            module.RMSNorm,
                            chatglm_rms_norm_forward)
        elif hasattr(model.config, 'vocab_size') and model.config.vocab_size == 130528:
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
            convert_forward(model,
                            module.RMSNorm,
                            llama_rms_norm_forward)
        elif model.config.hidden_size == 5120:
            # baichuan2-13B
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.baichuan2 import baichuan_attention_forward_13b
            from bigdl.llm.transformers.models.baichuan2 import baichuan_13b_rms_norm_forward
            convert_forward(model,
                            module.BaichuanAttention,
                            baichuan_attention_forward_13b
                            )
            # baichuan2-13B's RMSNorm is a little different
            convert_forward(model,
                            module.RMSNorm,
                            baichuan_13b_rms_norm_forward)
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
            convert_forward(model,
                            module.RMSNorm,
                            llama_rms_norm_forward)
        elif model.config.hidden_size == 5120:
            # baichuan-13B
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.baichuan import baichuan_attention_forward_13b
            from bigdl.llm.transformers.models.baichuan2 import baichuan_13b_rms_norm_forward
            convert_forward(model,
                            module.BaichuanAttention,
                            baichuan_attention_forward_13b
                            )
            # baichuan-13B's RMSNorm is a little different
            convert_forward(model,
                            module.RMSNorm,
                            baichuan_13b_rms_norm_forward)
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
        convert_forward(model,
                        module.InternLMRMSNorm,
                        llama_rms_norm_forward
                        )
    elif model.config.model_type == "qwen":
        if hasattr(model.config, "visual"):
            # for Qwen-VL-Chat
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from bigdl.llm.transformers.models.qwen_vl import qwen_attention_forward_vl
            convert_forward(model,
                            module.QWenAttention,
                            qwen_attention_forward_vl
                            )
        else:
            # for Qwen-7B and Qwen-14B
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
        convert_forward(model,
                        module.AquilaRMSNorm,
                        llama_rms_norm_forward)
    elif model.config.model_type == "mistral":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.mistral import mistral_attention_forward
        convert_forward(model,
                        module.MistralAttention,
                        mistral_attention_forward
                        )
        convert_forward(model,
                        module.MistralRMSNorm,
                        llama_rms_norm_forward)
    return model

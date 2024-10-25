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
import torch.distributed
import torch.nn as nn
from accelerate import init_empty_weights
import warnings
import transformers
import importlib.util
from ipex_llm.ggml.quantize import ggml_tensor_qtype, gguf_mixed_qtype
from .utils import logger, get_cur_qtype_and_imatrix, check_hidden_size
import numpy as np
import os
from ipex_llm.utils.common import invalidInputError
from typing import List, Optional, Tuple, Union
from types import MethodType
import subprocess
import sys

_IS_VLLM_AVAILABLE = None
_USE_VLLM = False
_USE_VLLM_AWQ = False
_USE_VLLM_GPTQ = False
_VLLM_VERSION = None


def is_auto_gptq_available():
    return importlib.util.find_spec("auto_gptq") is not None


def is_auto_awq_available():
    return importlib.util.find_spec("awq") is not None


def is_vllm_available():
    global _IS_VLLM_AVAILABLE
    if _IS_VLLM_AVAILABLE is not None:
        return _IS_VLLM_AVAILABLE
    import sys
    original_path = sys.path
    # Temporally remove current directory
    sys.path = original_path[1:]
    _IS_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
    sys.path = original_path
    return _IS_VLLM_AVAILABLE


def get_package_version(package_name):
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith(package_name):
            return line.split()[1]
    return None


def get_use_vllm():
    return _USE_VLLM


def is_torch_distributed_initialized():
    return torch.distributed.is_initialized()


def is_module_in_classes(module, classes):
    return any(isinstance(module, cls) for cls in classes)


def is_deepspeed_available():
    spec = importlib.util.find_spec("deepspeed")
    if spec is not None:
        deepspeed_path = spec.submodule_search_locations[0]
        if deepspeed_path != os.path.join(os.getcwd(), "deepspeed"):
            return True
        else:
            # not deepspeed package, just local dir
            return False
    else:
        return False


if is_auto_gptq_available():
    from auto_gptq.utils.peft_utils import QuantLinearCuda, QuantLinearCudaOld

if is_auto_awq_available():
    from ipex_llm.transformers.awq.linear import WQLinear_GEMM
    from transformers.utils.quantization_config import AwqBackendPackingMethod


def is_lm_head(name, model_config, out_features):
    if name == "lm_head" or getattr(model_config, "vocab_size", None) == out_features:
        return True
    else:
        return False


def is_gptq_linear(module):
    return is_auto_gptq_available() and \
        (isinstance(module, QuantLinearCuda) or isinstance(module, QuantLinearCudaOld))


def is_linear_module(module):

    global _USE_VLLM

    in_features = None
    out_features = None
    mp_group = None

    is_awq = is_auto_awq_available() and isinstance(module, WQLinear_GEMM)
    if is_vllm_available():
        # Only convert vllm modules
        global _VLLM_VERSION, _USE_VLLM_AWQ, _USE_VLLM_GPTQ
        if _VLLM_VERSION is None:
            _VLLM_VERSION = get_package_version('vllm')
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, RowParallelLinear, QKVParallelLinear, MergedColumnParallelLinear
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        VLLM_LINEAR_LIST = [
            ColumnParallelLinear, RowParallelLinear, QKVParallelLinear,
            MergedColumnParallelLinear,
        ]
        if 'xpu' in _VLLM_VERSION:
            VLLM_LINEAR_LIST.append(ParallelLMHead)
        if is_module_in_classes(module, VLLM_LINEAR_LIST):
            if 'xpu' in _VLLM_VERSION:
                # For vllm xpu
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_group,
                    get_tensor_model_parallel_world_size
                )
                if torch.distributed.is_initialized():
                    tp_size = get_tensor_model_parallel_world_size()
                else:
                    tp_size = 1
            else:
                # For vllm cpu
                tp_size = 1
            if isinstance(module, ParallelLMHead) and 'xpu' in _VLLM_VERSION:
                in_features = module.embedding_dim
                out_features = module.num_embeddings_per_partition
                result = True
                mp_group = None
                return result, (in_features, out_features, mp_group)
            in_features = module.input_size
            out_features = module.output_size
            result = True
            mp_group = None
            # Check for attribute qweight
            if (not _USE_VLLM_AWQ
               and hasattr(module.quant_method, "quant_config")
               and module.quant_method.quant_config.get_name() == "awq"):
                _USE_VLLM_AWQ = True
            if (not _USE_VLLM_GPTQ
               and hasattr(module.quant_method, "quant_config")
               and module.quant_method.quant_config.get_name() == "gptq"):
                _USE_VLLM_GPTQ = True
            invalidInputError(module.skip_bias_add is not True, "Currently, ipex-vllm does not"
                              " support linear layers with skip_bias_add argument")
            if isinstance(module, RowParallelLinear) and tp_size >= 2:
                mp_group = get_tensor_model_parallel_group()
                in_features = module.input_size_per_partition
            elif isinstance(module, ColumnParallelLinear) and tp_size >= 2:
                out_features = module.output_size_per_partition
            _USE_VLLM = True
            return result, (in_features, out_features, mp_group)
    if is_gptq_linear(module):
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


def convert_vllm(module, qtype, in_features, out_features, mp_group, cur_qtype,
                 enable_xetla, optimize_lm_head, enable_scale_search):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from ipex_llm.transformers.low_bit_linear import LowBitLinear, \
        FP16Linear, BF16Linear, vLLMLowBitLinear, vLLMFP16Linear, vLLMBF16Linear
    optimize_lm_head = False
    if isinstance(module, ParallelLMHead):
        if qtype == ggml_tensor_qtype["fp16"]:
            new_linear = FP16Linear(
                in_features,
                out_features,
                module.bias is not None,
                mp_group=mp_group,
                optimize_lm_head=optimize_lm_head
            )
        elif qtype == ggml_tensor_qtype["bf16"]:
            new_linear = BF16Linear(
                in_features,
                out_features,
                module.bias is not None,
                mp_group=mp_group,
                optimize_lm_head=optimize_lm_head
            )
        else:
            new_linear = LowBitLinear(
                in_features,
                out_features,
                cur_qtype,
                module.bias is not None,
                mp_group=mp_group,
                enable_xetla=enable_xetla,
                optimize_lm_head=optimize_lm_head,
                enable_scale_search=enable_scale_search,
            )
    else:
        if qtype == ggml_tensor_qtype["fp16"]:
            new_linear = vLLMFP16Linear(
                in_features,
                out_features,
                module.bias is not None,
                mp_group=mp_group,
                optimize_lm_head=optimize_lm_head
            )
        elif qtype == ggml_tensor_qtype["bf16"]:
            new_linear = vLLMBF16Linear(
                in_features,
                out_features,
                module.bias is not None,
                mp_group=mp_group,
                optimize_lm_head=optimize_lm_head
            )
        else:
            new_linear = vLLMLowBitLinear(
                in_features,
                out_features,
                cur_qtype,
                module.bias is not None,
                mp_group=mp_group,
                enable_xetla=enable_xetla,
                optimize_lm_head=optimize_lm_head,
                enable_scale_search=enable_scale_search,
            )
    return new_linear


def convert_vllm_awq_or_gptq(module, gptq=False, act_order=False):
    from ipex_llm.transformers.low_bit_linear import get_block_size
    Q4_1 = get_block_size("asym_int4")

    scales = module.scales
    # vLLM only supports load 4-bits model, so this has been checked
    if gptq:
        bits = module.quant_method.quant_config.weight_bits
        wf = (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
                           dtype=torch.int32) * 4).unsqueeze(0)
    else:
        bits = 4
        wf = (torch.tensor([0, 4, 1, 5, 2, 6, 3, 7],
                           dtype=torch.int32) * 4).unsqueeze(0)
    group_size = module.quant_method.quant_config.group_size
    if int(group_size) % Q4_1 != 0:
        invalidInputError(False, (f"group_size:{group_size} must be divisible by "f"{Q4_1}."))

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(module.qzeros, 2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
    zeros = torch.bitwise_and(zeros, (2 ** bits) - 1)

    g_id_map = None

    if gptq:
        zeros = zeros + 1
    zeros = zeros.reshape(scales.shape)

    if not gptq:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(module.qweight, 2).expand(-1, -1, 32 // bits),
            wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2 ** bits) - 1)
        weight = weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])
    else:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(module.qweight, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1)).to(torch.int8)
        weight = torch.bitwise_and(weight, (2 ** bits) - 1)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if act_order:
            invalidInputError(module.g_idx.shape[0] == weight.shape[0],
                              "g_idx and weight shape mismatch")
            _, g_id_map = torch.sort(module.g_idx)
            weight = weight[g_id_map, :]

    # convert weight to ggml format
    weight = weight.reshape(weight.shape[0]//group_size, group_size, weight.shape[1])
    weight = weight.permute(2, 0, 1).reshape(weight.shape[2], -1, 2, Q4_1//2)
    weight = weight.transpose(2, 3)
    weight = torch.bitwise_left_shift(weight,
                                      torch.tensor([0, 4], dtype=torch.int8).reshape(1, 1, 1, 2))
    weight = torch.bitwise_or(weight[:, :, :, 0], weight[:, :, :, 1]).contiguous()

    # convert zeros to ggml format
    zeros = zeros.reshape(-1, 1, zeros.shape[1]).permute(2, 0, 1)\
        .unsqueeze(2)\
        .expand(-1, -1, group_size//Q4_1, -1)\
        .reshape(zeros.shape[1], -1, 1)\
        .contiguous().to(torch.float16)

    # convert scales to ggml format
    scales = scales.reshape(-1, 1, scales.shape[1]).permute(2, 0, 1)\
        .unsqueeze(2)\
        .expand(-1, -1, group_size//Q4_1, -1)\
        .reshape(scales.shape[-1], -1, 1)\
        .contiguous().to(torch.float16)

    m = -(zeros * scales)
    d = scales

    ggml_weight = torch.cat([d.view(torch.uint8),
                             m.view(torch.uint8),
                             weight.view(torch.uint8)], dim=-1)
    ggml_weight = ggml_weight.reshape([-1])

    return ggml_weight, g_id_map


def convert_gptq(module, awq=False, llm_awq=False, act_order=False):
    from ipex_llm.transformers.low_bit_linear import get_block_size
    Q4_1 = get_block_size("asym_int4")

    scales = module.scales

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(module.qzeros, 2).expand(-1, -1, 32 // module.bits),
        module.wf.unsqueeze(0)).to(torch.int16 if module.bits == 8 else torch.int8)
    zeros = torch.bitwise_and(zeros, (2 ** module.bits) - 1)

    g_id_map = None

    if not awq:
        zeros = zeros + 1
    zeros = zeros.reshape(scales.shape)

    if awq:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(module.qweight, 2).expand(-1, -1, 32 // module.bits),
            module.wf.unsqueeze(0)).to(torch.int16 if module.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2 ** module.bits) - 1)
        weight = weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])
        if llm_awq:
            weight = weight.t()
    else:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(module.qweight, 1).expand(-1, 32 // module.bits, -1),
            module.wf.unsqueeze(-1)).to(torch.int8)
        weight = torch.bitwise_and(weight, (2 ** module.bits) - 1)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if act_order:
            invalidInputError(module.g_idx.shape[0] == weight.shape[0],
                              "g_idx and weight shape mismatch")
            _, g_id_map = torch.sort(module.g_idx)
            weight = weight[g_id_map, :]

    # convert weight to ggml format
    weight = weight.reshape(weight.shape[0]//module.group_size, module.group_size, weight.shape[1])
    weight = weight.permute(2, 0, 1).reshape(weight.shape[2], -1, 2, Q4_1//2)
    weight = weight.transpose(2, 3)
    weight = torch.bitwise_left_shift(weight,
                                      torch.tensor([0, 4], dtype=torch.int8).reshape(1, 1, 1, 2))
    weight = torch.bitwise_or(weight[:, :, :, 0], weight[:, :, :, 1]).contiguous()

    # convert zeros to ggml format
    if llm_awq:
        real_scale_num = module.in_features // module.group_size
        zeros = zeros[:, : real_scale_num]
        scales = scales[:, : real_scale_num]
        zeros = zeros.t()
        scales = scales.t()
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

    return ggml_weight, g_id_map


def use_scale_search(model_config, qtype):
    if qtype == ggml_tensor_qtype["fp6"] and model_config.model_type not in ["qwen2"]:
        return True
    elif qtype == ggml_tensor_qtype["fp8_e4m3"] and \
            model_config.model_type not in ["qwen2", "baichuan"]:
        if model_config.model_type == "llama" and model_config.vocab_size == 128256 and \
                "instruct" in model_config._name_or_path.lower():
            # Llama-3-instruct
            return False
        return True
    return False


def _replace_with_low_bit_linear(model, qtype, modules_to_not_convert=None,
                                 convert_shape_only=False,
                                 cpu_embedding=False,
                                 prefix_name='',
                                 imatrix_data=None, embedding_qtype=None,
                                 model_config=None, torch_dtype=torch.float32,
                                 enable_xetla=False,
                                 mixed_precision=False,
                                 act_order=False,
                                 enable_scale_search=False,
                                 ):
    from ipex_llm.transformers.low_bit_linear import LowBitLinear, FP4Params, \
        FP16Linear, BF16Linear
    from ipex_llm.transformers.embedding import CPUEmbedding, DiskEmbedding, LowBitEmbedding
    has_been_replaced = False
    global _USE_VLLM_AWQ, _USE_VLLM_GPTQ

    for name, module in model.named_children():
        is_linear, linear_args = is_linear_module(module)
        full_module_name = prefix_name + '.' + name if prefix_name != '' else name

        # use sub-string to match, it may match `10` if user only pass a number like `0`
        if any(key in full_module_name for key in modules_to_not_convert):
            continue

        if is_linear and getattr(model_config, "model_type", None) == "chatglm" and \
                name == "lm_head":
            # Now we re-reference it to output_layer
            model._modules[name] = model._modules["transformer"]._modules["output_layer"]
            continue

        if is_linear and not isinstance(module, LowBitLinear):
            in_features, out_features, mp_group = linear_args
            optimize_lm_head = (
                is_lm_head(name, model_config, out_features)
                and (
                    not os.environ.get("IPEX_LLM_LAST_LM_HEAD", None) == "0"
                )
                and (
                    not (getattr(model_config, "model_type", "") == "baichuan" and
                         model.config.hidden_size == 5120)  # except baichuan2-13B
                )
            )
            with init_empty_weights():
                new_linear = None
                is_gptq = is_gptq_linear(module)
                is_awq = is_auto_awq_available() and isinstance(module, WQLinear_GEMM)
                is_llm_awq = is_awq and module.backend == AwqBackendPackingMethod.LLMAWQ
                if is_gptq or is_awq:
                    has_bias = module.bias is not None and module.bias.abs().sum() != 0
                    new_linear = LowBitLinear(
                        in_features,
                        out_features,
                        qtype=qtype,
                        bias=has_bias,
                        mp_group=mp_group,
                        enable_xetla=enable_xetla,
                        optimize_lm_head=optimize_lm_head,
                        act_order=act_order,
                        enable_scale_search=enable_scale_search,
                    )
                    device = module.qweight.data.device
                    invalidInputError(device.type != "meta",
                                      "converting from meta device is not supported")
                    weight, g_idx_map = convert_gptq(module,
                                                     awq=is_awq,
                                                     llm_awq=is_llm_awq,
                                                     act_order=act_order)
                    if act_order:
                        new_linear.g_idx_map = g_idx_map
                    # Copy the weights
                    paramsLowBit = FP4Params(data=weight,
                                             requires_grad=False,
                                             quantized=True,
                                             _shape=(out_features, in_features),
                                             convert_shape_only=convert_shape_only,
                                             qtype=qtype,
                                             enable_xetla=enable_xetla,
                                             enable_scale_search=enable_scale_search).to(device)
                    new_linear._parameters['weight'] = paramsLowBit
                    if has_bias:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)
                elif _USE_VLLM_AWQ or _USE_VLLM_GPTQ:
                    # User load an AWQ quantized model from vLLM
                    from ipex_llm.transformers.low_bit_linear import vLLMLowBitLinear
                    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
                    has_bias = module.bias is not None and module.bias.abs().sum() != 0
                    if isinstance(module, ParallelLMHead):
                        new_linear = LowBitLinear(
                            in_features,
                            out_features,
                            qtype=qtype,
                            bias=has_bias,
                            mp_group=mp_group,
                            enable_xetla=enable_xetla,
                            optimize_lm_head=False,
                            act_order=act_order,
                            enable_scale_search=enable_scale_search,
                        )
                        device = module.weight.data.device
                        cur_qtype, cur_imatrix = get_cur_qtype_and_imatrix(qtype,
                                                                           full_module_name,
                                                                           imatrix_data,
                                                                           model_config)
                        # Copy the weights
                        paramsLowBit = FP4Params(data=module.weight.data,
                                                 requires_grad=False,
                                                 quantized=False,
                                                 _shape=None,
                                                 convert_shape_only=convert_shape_only,
                                                 qtype=cur_qtype,
                                                 imatrix=cur_imatrix,
                                                 in_features=in_features,
                                                 enable_xetla=enable_xetla,
                                                 enable_scale_search=enable_scale_search).to(device)
                    else:
                        new_linear = vLLMLowBitLinear(
                            in_features,
                            out_features,
                            qtype=qtype,
                            bias=has_bias,
                            mp_group=mp_group,
                            enable_xetla=enable_xetla,
                            optimize_lm_head=False,
                            act_order=act_order,
                            enable_scale_search=enable_scale_search,
                        )
                        device = module.qweight.data.device
                        invalidInputError(device.type != "meta",
                                          "converting from meta device is not supported")
                        weight, g_idx_map = convert_vllm_awq_or_gptq(module, gptq=_USE_VLLM_GPTQ,
                                                                     act_order=act_order)
                        if act_order:
                            new_linear.g_idx_map = g_idx_map
                        # Copy the weights
                        paramsLowBit = FP4Params(data=weight,
                                                 requires_grad=False,
                                                 quantized=True,
                                                 _shape=(out_features, in_features),
                                                 convert_shape_only=convert_shape_only,
                                                 qtype=qtype,
                                                 enable_xetla=enable_xetla,
                                                 enable_scale_search=enable_scale_search).to(device)
                    new_linear._parameters['weight'] = paramsLowBit
                    if has_bias:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)
                elif qtype not in [ggml_tensor_qtype["fp16"], ggml_tensor_qtype["bf16"]]:
                    if in_features % 64 != 0:
                        # now our kernel requires in_features is a multiple of 64
                        continue
                    cur_qtype, cur_imatrix = get_cur_qtype_and_imatrix(qtype,
                                                                       full_module_name,
                                                                       imatrix_data,
                                                                       model_config)
                    # mixed precison for lm_head
                    if mixed_precision and is_lm_head(name, model_config, out_features):
                        if cur_qtype in [ggml_tensor_qtype["sym_int4"],
                                         ggml_tensor_qtype["asym_int4"]]:
                            cur_qtype = ggml_tensor_qtype["q6_k"]

                    # check hidden size whether is a multiple of 256
                    cur_qtype = check_hidden_size(cur_qtype, in_features)

                    if _USE_VLLM:
                        new_linear = convert_vllm(module,
                                                  qtype,
                                                  in_features,
                                                  out_features,
                                                  mp_group,
                                                  cur_qtype,
                                                  enable_xetla,
                                                  optimize_lm_head,
                                                  enable_scale_search)
                    else:
                        new_linear = LowBitLinear(
                            in_features,
                            out_features,
                            cur_qtype,
                            module.bias is not None,
                            mp_group=mp_group,
                            enable_xetla=enable_xetla,
                            optimize_lm_head=optimize_lm_head,
                            enable_scale_search=enable_scale_search,
                        )
                    device = module.weight.data.device
                    # Copy the weights
                    paramsLowBit = FP4Params(data=module.weight.data,
                                             requires_grad=False,
                                             quantized=False,
                                             _shape=None,
                                             convert_shape_only=convert_shape_only,
                                             qtype=cur_qtype,
                                             imatrix=cur_imatrix,
                                             in_features=in_features,
                                             enable_xetla=enable_xetla,
                                             enable_scale_search=enable_scale_search).to(device)
                    new_linear._parameters['weight'] = paramsLowBit
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)
                elif qtype == ggml_tensor_qtype["fp16"]:
                    module.to(torch.float16)
                    if _USE_VLLM:
                        new_linear = convert_vllm(
                            module,
                            qtype,
                            in_features,
                            out_features,
                            mp_group,
                            None,
                            None,
                            optimize_lm_head,
                            None
                        )
                    else:
                        new_linear = FP16Linear(
                            in_features,
                            out_features,
                            module.bias is not None,
                            mp_group=mp_group,
                            optimize_lm_head=optimize_lm_head
                        )
                    device = module.weight.data.device
                    from ipex_llm.transformers.utils import get_ipex_version
                    if get_ipex_version() < "2.1.10+xpu":
                        new_linear._parameters['weight'] = nn.Parameter(module.weight)
                    else:
                        # only from 2.1, ipex provides matmul_bias_out
                        # so we need to transpose weight
                        new_weight = module.weight.transpose(0, 1).contiguous()
                        new_linear._parameters['weight'] = nn.Parameter(new_weight)
                        new_linear.weight_type = 2
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)
                elif qtype == ggml_tensor_qtype["bf16"]:
                    module.to(torch.bfloat16)
                    if _USE_VLLM:
                        new_linear = convert_vllm(
                            module,
                            qtype,
                            in_features,
                            out_features,
                            mp_group,
                            None,
                            None,
                            optimize_lm_head,
                            None
                        )
                    else:
                        new_linear = BF16Linear(
                            in_features,
                            out_features,
                            module.bias is not None,
                            mp_group=mp_group,
                            optimize_lm_head=optimize_lm_head
                        )
                    device = module.weight.data.device
                    # convert here
                    new_linear._parameters['weight'] = nn.Parameter(module.weight)
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)

                if new_linear is not None:
                    if not module.training:
                        new_linear.eval()
                    model._modules[name] = new_linear
                    has_been_replaced = True
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)

                    module.weight = None
        # skip user-defined Embedding layer
        elif cpu_embedding and type(module) == nn.Embedding:
            model._modules[name] = CPUEmbedding.from_embedding(module)
        elif embedding_qtype is not None and type(module) == nn.Embedding:
            model._modules[name] = LowBitEmbedding.from_embedding(module,
                                                                  convert_shape_only,
                                                                  embedding_qtype)
        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            _, _flag = _replace_with_low_bit_linear(
                module,
                qtype,
                modules_to_not_convert,
                convert_shape_only,
                cpu_embedding,
                prefix_name=prefix_name + '.' + name if prefix_name != '' else name,
                imatrix_data=imatrix_data,
                embedding_qtype=embedding_qtype,
                model_config=model_config,
                torch_dtype=torch_dtype,
                enable_xetla=enable_xetla,
                mixed_precision=mixed_precision,
                act_order=act_order,
                enable_scale_search=enable_scale_search,
            )
            has_been_replaced = _flag or has_been_replaced
    return model, has_been_replaced


def replace_with_low_bit_linear_for_module(model, qtype, module_name=None,
                                           modules_to_not_convert=None, current_key_name=None,
                                           convert_shape_only=False, torch_dtype="auto"):
    from ipex_llm.transformers.low_bit_linear import LowBitLinear, FP4Params, \
        FP16Linear, BF16Linear
    has_been_replaced = False

    splits = []
    if "." in module_name:
        splits = module_name.split(".")
    if not splits:
        invalidInputError(False,
                          "Please provide a valid module_name with hierarchical structure")
    else:
        parent_module = getattr(model, splits[0])

        if "lm_head" not in module_name:
            for split in splits[1:-2]:
                new_module = getattr(parent_module, split)
                parent_module = new_module
            module = getattr(parent_module, splits[-2])
            module_name = splits[-2]
        else:
            module = parent_module
            parent_module = model
            module_name = splits[0]

    if current_key_name is None:
        current_key_name = []

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    is_linear, linear_args = is_linear_module(module)
    if is_linear and module_name not in modules_to_not_convert:
        # Check if the current key is not in the `modules_to_not_convert`
        if (not any(key in ".".join(current_key_name) for key in modules_to_not_convert) and
                module.weight.data.device.type != 'meta' and not isinstance(module, LowBitLinear)):
            in_features, out_features, mp_group = linear_args
            with init_empty_weights():
                new_linear = None
                is_gptq = is_gptq_linear(module)
                is_awq = is_auto_awq_available() and isinstance(module, WQLinear_GEMM)
                is_llm_awq = is_awq and module.backend == AwqBackendPackingMethod.LLMAWQ
                if is_gptq or is_awq:
                    has_bias = module.bias is not None and module.bias.abs().sum() != 0
                    new_linear = LowBitLinear(
                        in_features,
                        out_features,
                        qtype=qtype,
                        bias=has_bias,
                        mp_group=mp_group,
                    )
                    device = module.qweight.data.device
                    invalidInputError(device.type != "meta",
                                      "converting from meta device is not supported")
                    # Copy the weights
                    paramsLowBit = FP4Params(data=convert_gptq(module, awq=is_awq,
                                                               llm_awq=is_llm_awq),
                                             requires_grad=False,
                                             quantized=True,
                                             _shape=(out_features, in_features),
                                             convert_shape_only=convert_shape_only,
                                             qtype=qtype).to(device)
                    new_linear._parameters['weight'] = paramsLowBit
                    if has_bias:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)
                elif qtype not in [ggml_tensor_qtype["fp16"], ggml_tensor_qtype["bf16"]]:
                    new_linear = LowBitLinear(
                        in_features,
                        out_features,
                        qtype,
                        module.bias is not None,
                        mp_group=mp_group,
                    )

                    device = module.weight.data.device
                    # Copy the weights
                    paramsLowBit = FP4Params(data=module.weight.data,
                                             requires_grad=False,
                                             quantized=False,
                                             _shape=None,
                                             convert_shape_only=convert_shape_only,
                                             qtype=qtype).to(device)
                    new_linear._parameters['weight'] = paramsLowBit
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)
                elif qtype == ggml_tensor_qtype["fp16"]:
                    module.to(torch.float16)
                    new_linear = FP16Linear(
                        in_features,
                        out_features,
                        module.bias is not None,
                        mp_group=mp_group,
                    )
                    device = module.weight.data.device
                    from ipex_llm.transformers.utils import get_ipex_version
                    if get_ipex_version() < "2.1.10+xpu":
                        new_linear._parameters['weight'] = nn.Parameter(module.weight)
                    else:
                        # only from 2.1, ipex provides matmul_bias_out
                        # so we need to transpose weight
                        new_weight = module.weight.transpose(0, 1).contiguous()
                        new_linear._parameters['weight'] = nn.Parameter(new_weight)
                        new_linear.weight_type = 2
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)
                elif qtype == ggml_tensor_qtype["bf16"]:
                    module.to(torch.bfloat16)
                    new_linear = BF16Linear(
                        in_features,
                        out_features,
                        module.bias is not None,
                        mp_group=mp_group,
                    )
                    device = module.weight.data.device
                    # convert here
                    new_linear._parameters['weight'] = nn.Parameter(module.weight)
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device)

                if new_linear is not None:
                    if not module.training:
                        new_linear.eval()
                    parent_module._modules[module_name] = new_linear
                    has_been_replaced = True
                    # Force requires grad to False to avoid unexpected errors
                    parent_module._modules[module_name].requires_grad_(False)

                    module.weight = None

    if has_been_replaced:
        if not (getattr(model, "quantization_method", None) == "gptq"):
            if torch_dtype == "auto":
                convert_bigdl_other_module(model, torch.float32)
            else:
                convert_bigdl_other_module(model, torch_dtype)
    return model


def _optimize_pre(model, qtype=None):
    try:
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        if isinstance(model, SentenceTransformer):
            if str(model._modules['0']).strip().split(' ')[-1] == 'BertModel':
                from ipex_llm.transformers.models.bert import merge_qkv
                model.apply(merge_qkv)
                return model
    except ModuleNotFoundError:
        pass

    from transformers.modeling_utils import PreTrainedModel
    # All huggingface format models are inherited from `PreTrainedModel`
    if not isinstance(model, PreTrainedModel):
        logger.info("Only HuggingFace Transformers models are currently "
                    "supported for further optimizations")
        return model
    # for rwkv models (verified RWKV/rwkv-4-world-7b)
    if model.config.model_type == "rwkv":
        model.rwkv._rescale_layers()
        model.rwkv.layers_are_rescaled = True
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

        # for baichuan2-7B
        if model.config.hidden_size in [4096, 2048]:
            from ipex_llm.transformers.models.baichuan import pre_compute_inv_freq
            model.apply(pre_compute_inv_freq)
    # for yuan 2.0
    if model.config.model_type == "yuan":
        from ipex_llm.transformers.models.yuan import merge_qk
        model.apply(merge_qk)
    # for bge-large
    if model.config.model_type == 'bert' and (
        not model.config.is_decoder and
        model.config.position_embedding_type == "absolute"
    ):
        from ipex_llm.transformers.models.bert import merge_qkv
        model.apply(merge_qkv)
    # for starcoder2
    if model.config.model_type == "starcoder2":
        from ipex_llm.transformers.models.starcoder2 import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type == "phi":
        from ipex_llm.transformers.models.phi import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type in ["phi3", "phi3_v"]:
        from ipex_llm.transformers.models.phi3 import pre_compute_inv_freq
        model.apply(pre_compute_inv_freq)
        from ipex_llm.transformers.models.phi3 import split_mlp
        model.apply(split_mlp)
    # for qwen2
    if model.config.model_type == "qwen2":
        # Skip merge_qkv and padding_mlp if quant_method is 'gptq'
        should_apply_merge_qkv = (
            not hasattr(model.config, "quantization_config") or
            not hasattr(model.config.quantization_config, "quant_method") or
            model.config.quantization_config.quant_method != "gptq"
        )
        if should_apply_merge_qkv:
            from ipex_llm.transformers.models.qwen2 import merge_qkv
            model.apply(merge_qkv)
            if qtype != ggml_tensor_qtype["fp6"]:
                from ipex_llm.transformers.models.qwen2 import padding_mlp
                model.apply(padding_mlp)
    if model.config.model_type == "qwen2_moe":
        from ipex_llm.transformers.models.qwen2_moe import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type == "qwen2_audio":
        from ipex_llm.transformers.models.qwen2 import merge_qkv
        model.language_model.apply(merge_qkv)
    if model.config.model_type == "qwen2_vl":
        from ipex_llm.transformers.models.qwen2_vl import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type == "stablelm":
        # For stablelm-zephyr-3b and stablelm-2-zephyr-1_6b
        from ipex_llm.transformers.models.stablelm import merge_qkv
        model.apply(merge_qkv)
    # for internlm
    if model.config.model_type == "internlm":
        from ipex_llm.transformers.models.internlm import merge_qkv
        model.apply(merge_qkv)
    # for internlm-xcomposer2-vl
    if model.config.model_type == "internlmxcomposer2":
        from ipex_llm.transformers.models.internlm import pre_process_attn_and_mlp
        model.apply(pre_process_attn_and_mlp)
    if model.config.model_type == "internvl_chat":
        _optimize_pre(model.language_model, qtype=qtype)
    if model.config.model_type == "gemma":
        from ipex_llm.transformers.models.gemma import merge_qkv, pre_compute_inv_freq
        model.apply(merge_qkv)
        model.apply(pre_compute_inv_freq)
    if model.config.model_type == "gemma2":
        from ipex_llm.transformers.models.gemma2 import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type == "llama":
        from ipex_llm.transformers.models.llama import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type == "mllama":
        from ipex_llm.transformers.models.mllama import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type == "minicpm":
        from ipex_llm.transformers.models.minicpm import merge_qkv
        model.apply(merge_qkv)
    if model.config.model_type == "minicpm3":
        from ipex_llm.transformers.models.minicpm3 import pre_compute_inv_freq
        model.apply(pre_compute_inv_freq)
        from ipex_llm.transformers.models.minicpm3 import padding_v_head_dim
        model.apply(padding_v_head_dim)
    if model.config.model_type == "minicpmv":
        from ipex_llm.transformers.models.minicpmv import merge_qkv
        model.vpm.apply(merge_qkv)
        if model.config.hidden_size == 2304 and model.config.vocab_size == 122753:
            model.llm.config.model_type = "minicpm"
        elif model.config.hidden_size == 3584 and model.config.vocab_size == 151666:
            model.llm.config.model_type = "qwen2"
        elif model.config.hidden_size == 4096 and model.config.vocab_size == 128256:
            model.llm.config.model_type = "llama"
        _optimize_pre(model.llm, qtype=qtype)
        model.llm.config.model_type = "minicpmv"

    return model


def ggml_convert_low_bit(model, qtype, optimize_model=True,
                         convert_shape_only=False, device="cpu",
                         modules_to_not_convert=None,
                         cpu_embedding=False,
                         lightweight_bmm=False, torch_dtype="auto",
                         imatrix_data=None,
                         embedding_qtype=None,
                         enable_xetla=False,
                         mixed_precision=False):
    if qtype in ggml_tensor_qtype.values():
        index = list(ggml_tensor_qtype.values()).index(qtype)
        logger.info(f"Converting the current model to "
                    f"{list(ggml_tensor_qtype.keys())[index]} "
                    f"format......")
    elif qtype in gguf_mixed_qtype.values():
        index = list(gguf_mixed_qtype.values()).index(qtype)
        logger.info(f"Converting the current model to "
                    f"{list(gguf_mixed_qtype.keys())[index]} "
                    f"format......")
    modules_to_not_convert = [] if modules_to_not_convert is None else modules_to_not_convert
    # Disable ipex duplicate import checker
    from ipex_llm.utils.ipex_importer import revert_import
    revert_import()

    # using ipex_llm optimizer before changing to bigdl linear
    _enable_ipex = get_enable_ipex()

    if _enable_ipex:
        model = _optimize_ipex(model, qtype)
        return model

    if optimize_model:
        model = _optimize_pre(model, qtype)

    act_order = False
    if getattr(model, "quantization_method", None) == "gptq":
        act_order = model.config.quantization_config.desc_act

    model_config = getattr(model, "config", None)

    enable_scale_search = use_scale_search(model_config, qtype)

    # mixed quantization needs model_config to choose custom quantization strategy
    if qtype is not None:
        model, has_been_replaced = _replace_with_low_bit_linear(
            model, qtype, modules_to_not_convert,
            convert_shape_only, cpu_embedding,
            imatrix_data=imatrix_data,
            embedding_qtype=embedding_qtype,
            model_config=model_config,
            torch_dtype=torch_dtype,
            enable_xetla=enable_xetla,
            mixed_precision=mixed_precision,
            act_order=act_order,
            enable_scale_search=enable_scale_search,
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
                if torch_dtype == "auto":
                    convert_bigdl_other_module(model, torch.float32)
                else:
                    convert_bigdl_other_module(model, torch_dtype)
        elif device == "meta":
            # Do nothing here for weights are empty.
            pass

    if optimize_model:
        model = _optimize_post(model, lightweight_bmm)

    if hasattr(model, "config") and hasattr(model.config, "model_type") and \
            model.config.model_type == "qwen" and hasattr(model.config, "visual"):
        # for Qwen-VL-Chat
        # Due to issue https://github.com/intel/intel-extension-for-pytorch/issues/454,
        # currently put interpolation execution into cpu
        visual_module_name = model.transformer.visual.__class__.__module__
        visual_module = importlib.import_module(visual_module_name)
        from ipex_llm.transformers.models.qwen_vl import qwen_vl_vision_transformer_forward
        from ipex_llm.transformers.models.qwen_vl import qwen_vl_resampler_forward
        convert_forward(model,
                        visual_module.VisionTransformer,
                        qwen_vl_vision_transformer_forward
                        )
        convert_forward(model,
                        visual_module.Resampler,
                        qwen_vl_resampler_forward
                        )
    return model


def convert_bigdl_other_module(model, dtype):
    # Convert modules outside of bigdl linear to corresponding dtype
    from ipex_llm.transformers.low_bit_linear import LowBitLinear, \
        FP16Linear, BF16Linear
    for module in model.modules():
        if list(module.children()) == []:
            # leaf module
            if not isinstance(module, (LowBitLinear, FP16Linear, BF16Linear)):
                module.to(dtype)


def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)


def replace_RotaryEmbed(m, target_m,  replace_embed):
    for attr_name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            setattr(m, attr_name, replace_embed(sub_m.dim,
                                                sub_m.max_position_embeddings,
                                                sub_m.base))
        replace_RotaryEmbed(sub_m, target_m, replace_embed)


def replace_func(m, target_m, func_name, new_func):
    for _, sub_m in m.named_children():
        if sub_m.__class__ == target_m:
            bound_method = new_func.__get__(sub_m, sub_m.__class__)
            setattr(sub_m, func_name, bound_method)
        replace_func(sub_m, target_m, func_name, new_func)


def get_enable_ipex():
    _enable_ipex = os.getenv("BIGDL_OPT_IPEX")
    _enable_ipex = (_enable_ipex is not None) and (_enable_ipex.lower() == "true")
    return _enable_ipex


def _optimize_ipex(model, qtype=ggml_tensor_qtype["bf16"]):
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.transformers.optimize import model_convert_reference
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from ipex_llm.transformers.convert_ipex import (
        _ipex_optimize_model, _ipex_jit, _make_causal_mask,
        _llama_model_forward_4_35, convert_function, GLM_get_masks,
    )

    model = model_convert_reference(model)

    rms_classes = [
        transformers.models.llama.modeling_llama.LlamaRMSNorm,
    ]
    if 'llama' in model.config.model_type:
        AttentionMaskConverter._make_causal_mask = _make_causal_mask
        convert_forward(model, transformers.models.llama.modeling_llama.LlamaModel,
                        _llama_model_forward_4_35)
    elif "mistral" in model.config.model_type:
        AttentionMaskConverter._make_causal_mask = _make_causal_mask
        convert_forward(model, transformers.models.llama.modeling_llama.LlamaModel,
                        _llama_model_forward_4_35)
    elif model.config.architectures is not None \
        and model.config.architectures[0] in ["ChatGLMModel", "ChatGLMForConditionalGeneration"]:  # noqa
        # for chatglm3-6B
        rms_classes.append(
            type(model.transformer.encoder.layers[0].input_layernorm)
        )
        convert_function(model.transformer, "get_masks", GLM_get_masks)
    elif model.config.model_type == 'baichuan' and model.config.vocab_size == 125696:
        # baichuan2
        rms_classes.append(type(model.model.layers[0].input_layernorm))

    model = _ipex_optimize_model(model, rms_classes, qtype)
    return _ipex_jit(model)


def _optimize_post(model, lightweight_bmm=False):
    try:
        from diffusers import StableDiffusionPipeline
        if isinstance(model, StableDiffusionPipeline):
            from ipex_llm.transformers.models.sd15 import AttnProcessor2_0
            model.unet.set_attn_processor(AttnProcessor2_0())
            return model
    except ModuleNotFoundError:
        pass

    try:
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        if isinstance(model, SentenceTransformer):
            if str(model._modules['0']).strip().split(' ')[-1] == 'BertModel':
                modeling_module_name = model._modules['0'].auto_model.__class__.__module__
                module = importlib.import_module(modeling_module_name)
                from ipex_llm.transformers.models.bert import self_attention_forward
                from ipex_llm.transformers.models.bert import encoder_forward
                convert_forward(model,
                                module.BertSelfAttention,
                                self_attention_forward)
                convert_forward(model,
                                module.BertEncoder,
                                encoder_forward)
                return model
    except ModuleNotFoundError:
        pass

    from transformers.modeling_utils import PreTrainedModel
    # All huggingface format models are inherited from `PreTrainedModel`
    if not isinstance(model, PreTrainedModel):
        logger.info("Only HuggingFace Transformers models are currently "
                    "supported for further optimizations")
        return model

    from packaging import version
    trans_version = transformers.__version__

    # convert all nn.LayerNorm
    from ipex_llm.transformers.models.bloom import bloom_layer_norm_forward
    convert_forward(model,
                    nn.LayerNorm,
                    bloom_layer_norm_forward)
    from ipex_llm.transformers.models.llama import llama_rms_norm_forward
    from ipex_llm.transformers.models.llama import llama_mlp_forward

    if model.config.model_type == "llama" and model.config.rope_scaling is not None:
        # llama 3.2 & llama 3.1
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.common import rms_norm_forward
        from ipex_llm.transformers.models.common import mlp_silu_forward
        from ipex_llm.transformers.models.llama32 import llama_model_forward
        from ipex_llm.transformers.models.llama32 import llama_attention_forward
        convert_forward(model, module.LlamaRMSNorm, rms_norm_forward)
        convert_forward(model, module.LlamaMLP, mlp_silu_forward)
        convert_forward(model, module.LlamaModel, llama_model_forward)
        convert_forward(model, module.LlamaAttention, llama_attention_forward)
        convert_forward(model, module.LlamaSdpaAttention, llama_attention_forward)
    elif model.config.model_type == "mllama":
        # llama 3.2 vision
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.mllama import mllama_vision_attention_forward
        convert_forward(model, module.MllamaVisionAttention, mllama_vision_attention_forward)
        convert_forward(model, module.MllamaVisionSdpaAttention, mllama_vision_attention_forward)

        from ipex_llm.transformers.models.common import rms_norm_forward
        from ipex_llm.transformers.models.common import mlp_silu_forward
        from ipex_llm.transformers.models.llama32 import llama_attention_forward
        from ipex_llm.transformers.models.mllama import mllama_text_model_forward
        from ipex_llm.transformers.models.mllama import mllama_cross_attention_forward
        convert_forward(model, module.MllamaTextRMSNorm, rms_norm_forward)
        convert_forward(model, module.MllamaTextMLP, mlp_silu_forward)
        convert_forward(model, module.MllamaTextModel, mllama_text_model_forward)
        convert_forward(model, module.MllamaTextSelfAttention, llama_attention_forward)
        convert_forward(model, module.MllamaTextSelfSdpaAttention, llama_attention_forward)
        convert_forward(model, module.MllamaTextCrossAttention, mllama_cross_attention_forward)
        convert_forward(model, module.MllamaTextCrossSdpaAttention, mllama_cross_attention_forward)
    elif model.config.model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        from transformers.models.llama.modeling_llama import LlamaMLP
        from transformers.models.llama.modeling_llama import LlamaAttention
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.llama.modeling_llama import LlamaModel
        if version.parse(trans_version) >= version.parse("4.36.0"):
            from transformers.models.llama.modeling_llama import LlamaSdpaAttention

        from ipex_llm.transformers.models.llama import llama_rms_norm_forward
        from ipex_llm.transformers.models.llama import llama_mlp_forward
        from ipex_llm.transformers.models.llama import llama_decoder_forward

        convert_forward(model, LlamaRMSNorm, llama_rms_norm_forward)
        convert_forward(model, LlamaMLP, llama_mlp_forward)
        convert_forward(model, LlamaDecoderLayer, llama_decoder_forward)

        if version.parse(trans_version) >= version.parse("4.41.0"):
            from ipex_llm.transformers.models.llama import llama_model_forward_4_41
            from ipex_llm.transformers.models.llama import llama_attention_forward_4_41
            convert_forward(model, LlamaModel, llama_model_forward_4_41)
            convert_forward(model, LlamaAttention, llama_attention_forward_4_41)
            convert_forward(model, LlamaSdpaAttention, llama_attention_forward_4_41)
        elif version.parse(trans_version) >= version.parse("4.38.0"):
            from ipex_llm.transformers.models.llama import llama_model_forward_4_38
            from ipex_llm.transformers.models.llama import llama_attention_forward_4_38
            convert_forward(model, LlamaModel, llama_model_forward_4_38)
            convert_forward(model, LlamaAttention, llama_attention_forward_4_38)
            convert_forward(model, LlamaSdpaAttention, llama_attention_forward_4_38)
        elif version.parse(trans_version) >= version.parse("4.36.0"):
            from ipex_llm.transformers.models.llama import llama_model_forward_4_36
            from ipex_llm.transformers.models.llama import llama_attention_forward_4_38
            convert_forward(model, LlamaModel, llama_model_forward_4_36)
            convert_forward(model, LlamaAttention, llama_attention_forward_4_38)
            convert_forward(model, LlamaSdpaAttention, llama_attention_forward_4_38)
        else:
            vllm_se_batching = os.getenv("VLLM_ENABLE_SELECTIVE_BATCHING", "").lower() == "true"
            if vllm_se_batching:
                from ipex_llm.transformers.models.llama import (
                    llama_model_selective_batching_forward_4_31,
                    llama_attention_selective_batching_forward_4_31,
                )
                convert_forward(model, LlamaModel,
                                llama_model_selective_batching_forward_4_31)
                convert_forward(model, LlamaAttention,
                                llama_attention_selective_batching_forward_4_31)
            else:
                from ipex_llm.transformers.models.llama import llama_model_forward
                from ipex_llm.transformers.models.llama import llama_attention_forward_4_31
                convert_forward(model, LlamaModel, llama_model_forward)
                convert_forward(model, LlamaAttention, llama_attention_forward_4_31)

    elif (
        model.config.architectures is not None
        and model.config.architectures[0] in ["ChatGLMModel", "ChatGLMForConditionalGeneration"]
    ):
        if hasattr(model.config, 'padded_vocab_size') and \
                model.config.padded_vocab_size in [65024, 64896]:
            # chatglm2-6b, chatglm2-6b-32k, chatglm3-6b, chatglm3-6b-32k, chatglm3-6b-128k
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from ipex_llm.transformers.models.chatglm2 import chatglm2_attention_forward
            from ipex_llm.transformers.models.chatglm2 import chatglm_rms_norm_forward
            from ipex_llm.transformers.models.chatglm2 import chatglm2_encoder_forward
            from ipex_llm.transformers.models.chatglm2 import chatglm2_model_forward
            convert_forward(model,
                            module.SelfAttention,
                            chatglm2_attention_forward)
            convert_forward(model,
                            module.GLMTransformer,
                            chatglm2_encoder_forward)
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
            from ipex_llm.transformers.models.chatglm import chatglm_attention_forward
            convert_forward(model,
                            module.SelfAttention,
                            chatglm_attention_forward
                            )
        elif model.config.num_layers == 40 and hasattr(model.config, 'rope_ratio'):
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            if hasattr(model.transformer, "vision"):
                # glm-4v-9b
                modeling_module_name = model.transformer.vision.__class__.__module__
                vision_module = importlib.import_module(modeling_module_name)
                from ipex_llm.transformers.models.chatglm4v import chatglm4v_attention_forward
                from ipex_llm.transformers.models.chatglm4v import chatglm4v_model_forward
                from ipex_llm.transformers.models.chatglm4v import visual_attention_forward
                from ipex_llm.transformers.models.chatglm4v import patch_embedding_forward
                from ipex_llm.transformers.models.chatglm2 import chatglm_rms_norm_forward
                convert_forward(model,
                                module.SelfAttention,
                                chatglm4v_attention_forward)
                convert_forward(model,
                                module.ChatGLMModel,
                                chatglm4v_model_forward)
                convert_forward(model,
                                module.RMSNorm,
                                chatglm_rms_norm_forward)
                convert_forward(model,
                                vision_module.Attention,
                                visual_attention_forward)
                convert_forward(model,
                                vision_module.PatchEmbedding,
                                patch_embedding_forward)
            else:
                # glm-4-9b-chat
                from ipex_llm.transformers.models.chatglm4 import chatglm4_attention_forward
                from ipex_llm.transformers.models.chatglm4 import chatglm4_model_forward
                from ipex_llm.transformers.models.chatglm2 import chatglm_rms_norm_forward
                from ipex_llm.transformers.models.chatglm4 import chatglm4_encoder_forward
                convert_forward(model,
                                module.SelfAttention,
                                chatglm4_attention_forward)
                convert_forward(model,
                                module.ChatGLMModel,
                                chatglm4_model_forward)
                convert_forward(model,
                                module.RMSNorm,
                                chatglm_rms_norm_forward)
                convert_forward(model,
                                module.GLMTransformer,
                                chatglm4_encoder_forward)

    elif "mpt" in model.config.model_type:
        if model.config.architectures is not None:
            modeling_module_name = model.__class__.__module__
            attention_module_name = '.'.join(modeling_module_name.split('.')[:-1]) + ".attention"
            module = importlib.import_module(attention_module_name)
            from ipex_llm.transformers.models.mpt import mpt_multihead_attention_forward
            convert_forward(model,
                            module.MultiheadAttention,
                            mpt_multihead_attention_forward
                            )
    elif "gptj" in model.config.model_type:
        # dolly-v1-6b
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.gptj import gptj_attention_forward, gptj_model_forward,\
            gptj_block_forward
        convert_forward(model,
                        module.GPTJAttention,
                        gptj_attention_forward)
        convert_forward(model,
                        module.GPTJModel,
                        gptj_model_forward)
        convert_forward(model,
                        module.GPTJBlock,
                        gptj_block_forward)
    elif "bloom" in model.config.model_type:
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.bloom import bloom_attention_forward
        convert_forward(model,
                        module.BloomAttention,
                        bloom_attention_forward
                        )
    elif "falcon" in model.config.model_type or "RefinedWeb" in model.config.model_type:
        if model.config.architectures is not None:
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            if "RWForCausalLM" in model.config.architectures:
                if model.config.hidden_size == 4544:
                    # falcon-7b need to check performance drop after kv cache support.
                    # from ipex_llm.transformers.models.falcon import rw_attention_forward_7b
                    # convert_forward(model,
                    #                 module.Attention,
                    #                 rw_attention_forward_7b
                    #                 )
                    pass
                else:
                    # falcon-40b
                    from ipex_llm.transformers.models.falcon import rw_attention_forward_40b
                    convert_forward(model,
                                    module.Attention,
                                    rw_attention_forward_40b
                                    )
            elif "FalconForCausalLM" in model.config.architectures:
                if model.config.hidden_size != 4544:
                    # falcon-180b and new falcon-40b
                    if version.parse(trans_version) >= version.parse("4.36.0"):
                        # transformers version >= 4.36.0
                        from ipex_llm.transformers.models.falcon import \
                            falcon_attention_forward_4_36

                        convert_forward(model,
                                        module.FalconAttention,
                                        falcon_attention_forward_4_36
                                        )
                    else:
                        from ipex_llm.transformers.models.falcon import falcon_attention_forward
                        convert_forward(model,
                                        module.FalconAttention,
                                        falcon_attention_forward
                                        )
    elif model.config.model_type == "baichuan":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.baichuan import baichuan_mlp_forward
        convert_forward(model, module.MLP, baichuan_mlp_forward)

        if model.config.hidden_size in [4096, 2048]:
            # baichuan-7B and baichuan2-7B
            from ipex_llm.transformers.models.baichuan import baichuan_attention_forward_7b
            from ipex_llm.transformers.models.baichuan import baichuan_model_7b_forward
            for i in range(len(model.model.layers)):
                setattr(model.model.layers[i].self_attn, "layer_idx", i)
            convert_forward(model, module.Attention, baichuan_attention_forward_7b)
            convert_forward(model, module.RMSNorm, llama_rms_norm_forward)
            if model.config.vocab_size == 125696:
                # baichuan2-7B
                convert_forward(model, module.BaichuanModel, baichuan_model_7b_forward)
            elif model.config.vocab_size == 64000:
                # baichuan-7B
                convert_forward(model, module.Model, baichuan_model_7b_forward)
        elif model.config.hidden_size == 5120:
            # baichuan-13B and baichuan2-13B
            from ipex_llm.transformers.models.baichuan import baichuan_attention_forward_13b
            from ipex_llm.transformers.models.baichuan import baichuan_13b_rms_norm_forward
            convert_forward(model, module.BaichuanAttention, baichuan_attention_forward_13b)
            convert_forward(model, module.RMSNorm, baichuan_13b_rms_norm_forward)

            if model.config.vocab_size == 125696:
                # baichaun2-13B
                from ipex_llm.transformers.models.baichuan import baichuan_13b_get_alibi_mask
                if hasattr(model.model, 'get_alibi_mask_orig'):
                    # deepspeed rewrite "get_alibi_mask" to support baichuan
                    # https://github.com/microsoft/DeepSpeed/pull/4721
                    replace_func(model,
                                 module.BaichuanModel,
                                 "get_alibi_mask_orig",
                                 baichuan_13b_get_alibi_mask)
                else:
                    replace_func(model,
                                 module.BaichuanModel,
                                 "get_alibi_mask",
                                 baichuan_13b_get_alibi_mask)
    elif model.config.model_type == "gpt2":
        from ipex_llm.transformers.models.gpt2 import gpt2_attention_attn
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        module.GPT2Attention._attn = gpt2_attention_attn
    elif model.config.model_type == "gpt_neox":
        from ipex_llm.transformers.models.gptneox import gptneox_attention_forward
        convert_forward(model,
                        transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention,
                        gptneox_attention_forward
                        )
    elif model.config.model_type == "internlm":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.internlm import internlm_attention_forward
        convert_forward(model, module.InternLMAttention, internlm_attention_forward)
        convert_forward(model, module.InternLMRMSNorm, llama_rms_norm_forward)
    elif model.config.model_type == "internlm2":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.internlm import internlm2_attention_forward
        convert_forward(model, module.InternLM2Attention, internlm2_attention_forward)
        convert_forward(model, module.InternLM2RMSNorm, llama_rms_norm_forward)
    elif model.config.model_type == "internlmxcomposer2":
        modeling_module_name = model.model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.internlm import (
            internlm_xcomposser2_attention_forward,
            internlm_xcomposser2_mlp_forward,
            internlm_xcomposser2_model_forward_wrapper,
            internlm_xcomposser2_chat
        )
        convert_forward(model, module.InternLM2Attention, internlm_xcomposser2_attention_forward)
        convert_forward(model, module.InternLM2MLP, internlm_xcomposser2_mlp_forward)
        convert_forward(model, module.InternLM2RMSNorm, llama_rms_norm_forward)
        internlm_xcomposser2_model_forward = internlm_xcomposser2_model_forward_wrapper(
            module.InternLM2Model.forward
        )
        convert_forward(model, module.InternLM2Model, internlm_xcomposser2_model_forward)
        model.chat = MethodType(internlm_xcomposser2_chat, model)
    elif model.config.model_type == "internvl_chat":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.internvl import internvl_chat
        from ipex_llm.transformers.models.internvl import internvl_batch_chat
        model.get_conv_template = module.get_conv_template
        model.chat = MethodType(internvl_chat, model)
        model.batch_chat = MethodType(internvl_batch_chat, model)
        if model.vision_model.__class__.__name__ == "InternVisionModel":
            from ipex_llm.transformers.models.internvl import _get_pos_embed
            from ipex_llm.transformers.models.internvl import intern_attention_forward
            vision_model = model.vision_model
            vision_embedding = vision_model.embeddings
            vision_embedding._get_pos_embed = MethodType(_get_pos_embed, vision_embedding)
            vision_module = importlib.import_module(vision_model.__class__.__module__)
            convert_forward(vision_model, vision_module.InternAttention, intern_attention_forward)
        _optimize_post(model.language_model, lightweight_bmm=lightweight_bmm)
    elif model.config.model_type == "qwen":
        if hasattr(model.config, "visual"):
            # for Qwen-VL-Chat
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from ipex_llm.transformers.models.qwen_vl import qwen_attention_forward_vl
            from ipex_llm.transformers.models.qwen_vl import qwen_vl_model_forward
            convert_forward(model,
                            module.QWenAttention,
                            qwen_attention_forward_vl
                            )
            convert_forward(model,
                            module.QWenModel,
                            qwen_vl_model_forward)
        else:
            # for Qwen-7B and Qwen-14B
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            from ipex_llm.transformers.models.qwen import qwen_attention_forward
            from ipex_llm.transformers.models.qwen import qwen_attention_forward_registered
            from ipex_llm.transformers.models.qwen import qwen_mlp_forward
            from ipex_llm.transformers.models.chatglm2 import chatglm_rms_norm_forward
            from ipex_llm.transformers.models.qwen import qwen_model_forward
            if model.config.max_position_embeddings == 8192 \
               and model.config.hidden_size == 4096:
                convert_forward(model,
                                module.QWenAttention,
                                qwen_attention_forward_registered
                                )
            else:
                convert_forward(model,
                                module.QWenAttention,
                                qwen_attention_forward
                                )
            convert_forward(model,
                            module.RMSNorm,
                            chatglm_rms_norm_forward)
            convert_forward(model,
                            module.QWenMLP,
                            qwen_mlp_forward)
            convert_forward(model,
                            module.QWenModel,
                            qwen_model_forward)
    elif model.config.model_type == "qwen2":
        # for Qwen1.5-7B
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.qwen2 import qwen2_model_forward
        from ipex_llm.transformers.models.qwen2 import qwen2_attention_forward
        from ipex_llm.transformers.models.qwen2 import qwen2_causal_lm_forward
        from ipex_llm.transformers.models.qwen2 import qwen2_mlp_forward
        convert_forward(model,
                        module.Qwen2ForCausalLM,
                        qwen2_causal_lm_forward)
        convert_forward(model,
                        module.Qwen2RMSNorm,
                        llama_rms_norm_forward)
        convert_forward(model,
                        module.Qwen2MLP,
                        qwen2_mlp_forward)
        convert_forward(model,
                        module.Qwen2Attention,
                        qwen2_attention_forward)
        convert_forward(model,
                        module.Qwen2SdpaAttention,
                        qwen2_attention_forward)
        if version.parse(trans_version) >= version.parse("4.42"):
            from ipex_llm.transformers.models.qwen2 import qwen2_model_forward_4_42
            convert_forward(model, module.Qwen2Model, qwen2_model_forward_4_42)
        else:
            from ipex_llm.transformers.models.qwen2 import qwen2_model_forward
            convert_forward(model, module.Qwen2Model, qwen2_model_forward)
    elif model.config.model_type == "qwen2_moe":
        # for Qwen1.5-MOE-A2.7B
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.qwen2_moe import qwen2moe_moeblock_forward
        from ipex_llm.transformers.models.qwen2_moe import qwen2moe_model_forward
        from ipex_llm.transformers.models.qwen2_moe import qwen2_moe_causal_lm_forward
        from ipex_llm.transformers.models.qwen2 import qwen2_attention_forward
        from ipex_llm.transformers.models.qwen2 import qwen2_mlp_forward
        convert_forward(model,
                        module.Qwen2MoeModel,
                        qwen2moe_model_forward)
        convert_forward(model,
                        module.Qwen2MoeForCausalLM,
                        qwen2_moe_causal_lm_forward)
        convert_forward(model,
                        module.Qwen2MoeRMSNorm,
                        llama_rms_norm_forward)
        convert_forward(model,
                        module.Qwen2MoeSparseMoeBlock,
                        qwen2moe_moeblock_forward)
        convert_forward(model,
                        module.Qwen2MoeMLP,
                        qwen2_mlp_forward)
        convert_forward(model,
                        module.Qwen2MoeAttention,
                        qwen2_attention_forward)
        convert_forward(model,
                        module.Qwen2MoeSdpaAttention,
                        qwen2_attention_forward)
    elif model.config.model_type == "qwen2_audio":
        _optimize_post(model.language_model, lightweight_bmm=lightweight_bmm)
    elif model.config.model_type == "qwen2_vl":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.common import rms_norm_forward
        from ipex_llm.transformers.models.qwen2 import qwen2_mlp_forward
        from ipex_llm.transformers.models.qwen2_vl import qwen2_vision_get_dtype
        from ipex_llm.transformers.models.qwen2_vl import qwen2_vision_attention_forward
        from ipex_llm.transformers.models.qwen2_vl import qwen2_vl_model_forward
        from ipex_llm.transformers.models.qwen2_vl import qwen2_vl_attention_forward
        convert_forward(model, module.Qwen2RMSNorm, rms_norm_forward)
        convert_forward(model, module.Qwen2MLP, qwen2_mlp_forward)
        model.visual.get_dtype = MethodType(qwen2_vision_get_dtype, model.visual)
        convert_forward(model, module.VisionAttention, qwen2_vision_attention_forward)
        convert_forward(model, module.Qwen2VLModel, qwen2_vl_model_forward)
        convert_forward(model, module.Qwen2VLAttention, qwen2_vl_attention_forward)
    elif model.config.model_type == "cohere":
        # for CohereForAI/c4ai-command-r-v01
        invalidInputError(version.parse(trans_version) >= version.parse("4.40.0"),
                          "Please upgrade transformers to 4.40.0 or higher version "
                          "to run Mixtral models.")
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        if version.parse(trans_version) >= version.parse("4.41.0"):
            from ipex_llm.transformers.models.cohere import cohere_model_forward_4_41
            convert_forward(model,
                            module.CohereModel,
                            cohere_model_forward_4_41)
        else:
            from ipex_llm.transformers.models.cohere import cohere_model_forward
            convert_forward(model,
                            module.CohereModel,
                            cohere_model_forward)

        from ipex_llm.transformers.models.cohere import cohere_attention_forward
        convert_forward(model,
                        module.CohereAttention,
                        cohere_attention_forward)
        convert_forward(model,
                        module.CohereLayerNorm,
                        llama_rms_norm_forward)
        convert_forward(model,
                        module.CohereMLP,
                        llama_mlp_forward)
    elif model.config.model_type == "aquila":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.aquila import aquila_attention_forward
        convert_forward(model,
                        module.AquilaAttention,
                        aquila_attention_forward
                        )
        convert_forward(model,
                        module.AquilaRMSNorm,
                        llama_rms_norm_forward)
    elif model.config.model_type == "mixtral":
        # For mistralai/Mixtral-8x7B-v0.1
        invalidInputError(version.parse(trans_version) >= version.parse("4.36.0"),
                          "Please upgrade transformers to 4.36.0 or higher version "
                          "to run Mixtral models.")
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.mixtral import mixtral_moeblock_forward, \
            mixtral_attention_forward, mixtral_mlp_forward, mixtral_model_forward
        convert_forward(model,
                        module.MixtralAttention,
                        mixtral_attention_forward)
        convert_forward(model,
                        module.MixtralRMSNorm,
                        llama_rms_norm_forward)
        convert_forward(model,
                        module.MixtralSparseMoeBlock,
                        mixtral_moeblock_forward)
        convert_forward(model,
                        module.MixtralBLockSparseTop2MLP,
                        mixtral_mlp_forward)
        convert_forward(model,
                        module.MixtralModel,
                        mixtral_model_forward)

    elif model.config.model_type == "phi-msft" and \
            hasattr(model.config, "num_local_experts"):
        # For phixtral, limit the condition to avoid applying on phi-2 hosted by ModelScope
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.phixtral import phixtral_moeblock_forward, \
            phixtral_mlp_forward
        convert_forward(model,
                        module.MoE,
                        phixtral_moeblock_forward)
        convert_forward(model,
                        module.MLP,
                        phixtral_mlp_forward)
    elif model.config.model_type == "mistral":
        if model.config.architectures is not None and \
                model.config.architectures[0] == "MixtralForCausalLM":
            # For DiscoResearch/mixtral-7b-8expert
            invalidInputError(version.parse(trans_version) >= version.parse("4.36.0"),
                              "Please upgrade transformers to 4.36.0 or higher version "
                              "to run Mixtral models.")
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            convert_forward(model,
                            module.MistralRMSNorm,
                            llama_rms_norm_forward)
        else:
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            if version.parse(trans_version) >= version.parse("4.36.0"):
                from ipex_llm.transformers.models.mistral import mistral_model_forward_4_36
                if version.parse(trans_version) >= version.parse("4.39.0"):
                    from ipex_llm.transformers.models.mistral import \
                        mistral_attention_forward_4_39
                    convert_forward(model,
                                    module.MistralAttention,
                                    mistral_attention_forward_4_39
                                    )
                else:
                    from ipex_llm.transformers.models.mistral import mistral_attention_forward_4_36
                    convert_forward(model,
                                    module.MistralAttention,
                                    mistral_attention_forward_4_36
                                    )
                convert_forward(model,
                                module.MistralModel,
                                mistral_model_forward_4_36
                                )
                convert_forward(model,
                                module.MistralRMSNorm,
                                llama_rms_norm_forward)
                convert_forward(model,
                                module.MistralMLP,
                                llama_mlp_forward)
            else:
                from ipex_llm.transformers.models.mistral import mistral_attention_forward
                convert_forward(model,
                                module.MistralAttention,
                                mistral_attention_forward
                                )
                convert_forward(model,
                                module.MistralRMSNorm,
                                llama_rms_norm_forward)
                convert_forward(model,
                                module.MistralMLP,
                                llama_mlp_forward)
    elif model.config.model_type == "gemma":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.gemma import gemma_model_forward
        from ipex_llm.transformers.models.gemma import gemma_attention_forward
        from ipex_llm.transformers.models.gemma import gemma_rms_norm_forward
        from ipex_llm.transformers.models.common import mlp_gelu_forward
        convert_forward(model, module.GemmaModel, gemma_model_forward)
        convert_forward(model, module.GemmaAttention, gemma_attention_forward)
        convert_forward(model, module.GemmaRMSNorm, gemma_rms_norm_forward)
        convert_forward(model, module.GemmaMLP, mlp_gelu_forward)
    elif model.config.model_type == "gemma2":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.common import mlp_gelu_forward
        from ipex_llm.transformers.models.gemma import gemma_rms_norm_forward
        from ipex_llm.transformers.models.gemma2 import gemma2_attention_forward
        from ipex_llm.transformers.models.gemma2 import gemma2_model_forward
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm, Gemma2Attention, \
            Gemma2SdpaAttention
        from transformers.models.gemma2.modeling_gemma2 import Gemma2Model, Gemma2MLP
        convert_forward(model, Gemma2RMSNorm, gemma_rms_norm_forward)
        convert_forward(model, Gemma2Attention, gemma2_attention_forward)
        convert_forward(model, Gemma2SdpaAttention, gemma2_attention_forward)
        convert_forward(model, Gemma2Model, gemma2_model_forward)
        convert_forward(model, Gemma2MLP, mlp_gelu_forward)
    elif model.config.model_type == "Yi":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        convert_forward(model,
                        module.YiRMSNorm,
                        llama_rms_norm_forward)
    elif model.config.model_type == "whisper" and lightweight_bmm:
        if platform.system().lower() == 'windows':
            from ipex_llm.transformers.bmm import SafeBMM
            modeling_module_name = model.__class__.__module__
            module = importlib.import_module(modeling_module_name)
            old_fwd = module.WhisperAttention.forward

            def safe_bmm_fwd(*args, **kwargs):
                with SafeBMM():
                    return old_fwd(*args, **kwargs)

            convert_forward(model,
                            module.WhisperAttention,
                            safe_bmm_fwd)
    elif model.config.model_type == "rwkv":
        # rwkv v4
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.rwkv4 import rwkv_attention_forward
        from ipex_llm.transformers.models.rwkv4 import rwkv_ffn_forward
        convert_forward(model,
                        module.RwkvSelfAttention,
                        rwkv_attention_forward)
        convert_forward(model,
                        module.RwkvFeedForward,
                        rwkv_ffn_forward)
    elif model.config.model_type == "rwkv5":
        # rwkv v5
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.rwkv5 import rwkv_attention_forward
        from ipex_llm.transformers.models.rwkv5 import rwkv_ffn_forward_wrapper
        from ipex_llm.transformers.models.rwkv5 import rwkv_model_forward_wrapper
        convert_forward(model,
                        module.RwkvSelfAttention,
                        rwkv_attention_forward)
        rwkv_ffn_forward = rwkv_ffn_forward_wrapper(module.RwkvFeedForward.forward)
        convert_forward(model,
                        module.RwkvFeedForward,
                        rwkv_ffn_forward)
        rwkv_model_forward = rwkv_model_forward_wrapper(module.Rwkv5Model.forward)
        convert_forward(model,
                        module.Rwkv5Model,
                        rwkv_model_forward)
    elif model.config.model_type == "deci":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.decilm import decilm_attention_forward_4_35_2
        convert_forward(model,
                        module.LlamaRMSNorm,
                        llama_rms_norm_forward)
        convert_forward(model,
                        module.LlamaMLP,
                        llama_mlp_forward)
        convert_forward(model,
                        module.DeciLMAttention,
                        decilm_attention_forward_4_35_2, )
    elif model.config.model_type == "gpt_bigcode":
        # starcoder
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.gptbigcode import _attn_wrapper
        from ipex_llm.transformers.models.gptbigcode import gptbigcode_attention_forward, \
            gptbigcode_sdpa_attention_forward
        convert_forward(model,
                        module.GPTBigCodeAttention,
                        gptbigcode_attention_forward)
        _attn = _attn_wrapper(module.GPTBigCodeAttention._attn)
        replace_func(model,
                     module.GPTBigCodeAttention,
                     "_attn",
                     _attn)
        try:
            # for transformers 4.36+
            convert_forward(model,
                            module.GPTBigCodeSdpaAttention,
                            gptbigcode_sdpa_attention_forward)
            sdpa_attn = _attn_wrapper(module.GPTBigCodeSdpaAttention._attn)
            replace_func(model,
                         module.GPTBigCodeSdpaAttention,
                         "_attn",
                         sdpa_attn)
        except AttributeError:
            pass
    elif model.config.model_type == "starcoder2":
        # starcoder2
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.starcoder2 import attention_forward
        from ipex_llm.transformers.models.starcoder2 import model_forward
        convert_forward(model, module.Starcoder2Attention, attention_forward)
        convert_forward(model, module.Starcoder2Model, model_forward)
    elif model.config.model_type == "phi":
        # for phi-2
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.phi import attention_forward
        from ipex_llm.transformers.models.phi import model_forward
        convert_forward(model, module.PhiAttention, attention_forward)
        convert_forward(model, module.PhiModel, model_forward)
    elif model.config.model_type in ["phi3", "phi3_v"]:
        # for phi-3
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.phi3 import attention_forward
        convert_forward(model, module.Phi3Attention, attention_forward)
        from ipex_llm.transformers.models.phi3 import mlp_forward
        convert_forward(model, module.Phi3MLP, mlp_forward)
        from ipex_llm.transformers.models.phi3 import phi3_rms_norm_forward
        convert_forward(model, module.Phi3RMSNorm, phi3_rms_norm_forward)
        if model.config.model_type == "phi3":
            from ipex_llm.transformers.models.phi3 import phi3_model_forward_wrapper
            model_forward = phi3_model_forward_wrapper(module.Phi3Model.forward)
            convert_forward(model, module.Phi3Model, model_forward)
        else:
            from ipex_llm.transformers.models.phi3 import phi3v_model_forward_wrapper
            model_forward = phi3v_model_forward_wrapper(module.Phi3VModel.forward)
            convert_forward(model, module.Phi3VModel, model_forward)
    elif model.config.model_type == 'yuan':
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.yuan import yuan_attention_forward
        # from ipex_llm.transformers.models.yuan import yuan_mlp_forward
        convert_forward(model,
                        module.YuanAttention,
                        yuan_attention_forward
                        )
        # disable able mlp_forward for quantize_kv on mtl.
        # convert_forward(model,
        #                 module.YuanMLP,
        #                 yuan_mlp_forward
        #                 )
    elif model.config.model_type == 'bert' and (
        not model.config.is_decoder and
        model.config.position_embedding_type == "absolute"
    ):
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.bert import self_attention_forward
        from ipex_llm.transformers.models.bert import encoder_forward
        convert_forward(model,
                        module.BertSelfAttention,
                        self_attention_forward)
        convert_forward(model,
                        module.BertEncoder,
                        encoder_forward)
    elif model.config.model_type == 'stablelm':
        # For stablelm-zephyr-3b and stablelm-2-zephyr-1_6b
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.stablelm import stablelm_attention_forward
        from ipex_llm.transformers.models.stablelm import stablelm_model_forward
        convert_forward(model,
                        module.StableLmAttention,
                        stablelm_attention_forward
                        )
        convert_forward(model,
                        module.StableLmMLP,
                        llama_mlp_forward)
        convert_forward(model,
                        module.StableLmModel,
                        stablelm_model_forward
                        )
    elif model.config.model_type == "minicpm":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.minicpm import minicpm_attention_forward
        from ipex_llm.transformers.models.minicpm import minicpm_model_forward_wrapper
        convert_forward(model, module.MiniCPMAttention, minicpm_attention_forward)
        convert_forward(model, module.MiniCPMMLP, llama_mlp_forward)
        convert_forward(model, module.MiniCPMRMSNorm, llama_rms_norm_forward)
        minicpm_model_forward = minicpm_model_forward_wrapper(module.MiniCPMModel.forward)
        convert_forward(model, module.MiniCPMModel, minicpm_model_forward)
    elif model.config.model_type == "minicpm3":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.common import rms_norm_forward
        from ipex_llm.transformers.models.common import mlp_silu_forward
        from ipex_llm.transformers.models.minicpm3 import minicpm3_attention_forward
        from ipex_llm.transformers.models.minicpm3 import minicpm3_model_forward_wrapper
        convert_forward(model, module.MiniCPMRMSNorm, rms_norm_forward)
        convert_forward(model, module.MiniCPMMLP, mlp_silu_forward)
        convert_forward(model, module.MiniCPMAttention, minicpm3_attention_forward)
        minicpm3_model_forward = minicpm3_model_forward_wrapper(module.MiniCPM3Model.forward)
        convert_forward(model, module.MiniCPM3Model, minicpm3_model_forward)
    elif model.config.model_type == "minicpmv":
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from ipex_llm.transformers.models.minicpmv import minicpmv_generate_wrapper
        minicpmv_generate = minicpmv_generate_wrapper(module.MiniCPMV.generate)
        model.generate = MethodType(minicpmv_generate, model)

        if model.config.hidden_size == 2304 and model.config.vocab_size == 122753:
            # MiniCPM-V 2
            model.llm.config.model_type = "minicpm"
        elif model.config.hidden_size == 3584 and model.config.vocab_size == 151666:
            # MiniCPM-V 2.6
            model.llm.config.model_type = "qwen2"
        elif model.config.hidden_size == 4096 and model.config.vocab_size == 128256:
            # MiniCPM-V 2.5
            model.llm.config.model_type = "llama"
        _optimize_post(model.llm, lightweight_bmm=lightweight_bmm)
        model.llm.config.model_type = "minicpmv"

        vpm_modeling_module_name = model.vpm.__class__.__module__
        vpm_module = importlib.import_module(vpm_modeling_module_name)
        if not hasattr(model.vpm, "config"):
            # MiniCPM-V 2
            from ipex_llm.transformers.models.minicpmv import vision_transformer_attention_forward
            from ipex_llm.transformers.models.minicpmv import minicpmv_get_vision_embedding
            convert_forward(model.vpm, vpm_module.Attention, vision_transformer_attention_forward)
            model.get_vision_embedding = MethodType(minicpmv_get_vision_embedding, model)
        elif "siglip" in model.vpm.config.model_type:
            # MiniCPM-V 2.6
            from ipex_llm.transformers.models.minicpmv import siglip_attention_forward
            convert_forward(model.vpm, vpm_module.SiglipAttention, siglip_attention_forward)

            from ipex_llm.transformers.models.minicpmv import _in_projection_packed
            resampler_module_name = model.resampler.__class__.__module__
            resampler_module = importlib.import_module(resampler_module_name)
            resampler_module._in_projection_packed = _in_projection_packed

            # for minicpm-v-2_6 benchmarking purposes
            from ipex_llm.transformers.models.minicpmv import minicpmv_decode_stream_wrapper
            minicpmv_decode_stream = minicpmv_decode_stream_wrapper(module.MiniCPMV._decode_stream)
            model._decode_stream = MethodType(minicpmv_decode_stream, model)
        elif model.vpm.config.model_type == "idefics2":
            # MiniCPM-V 2.5
            from ipex_llm.transformers.models.minicpmv import siglip_attention_forward
            from ipex_llm.transformers.models.minicpmv import minicpmv_chat_wrapper
            convert_forward(model.vpm, vpm_module.Idefics2VisionAttention, siglip_attention_forward)
            minicpmv_chat = minicpmv_chat_wrapper(module.MiniCPMV.chat)
            model.chat = MethodType(minicpmv_chat, model)

    return model

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


import auto_round_patch
import torch
import os
from ipex_llm.utils.common.log4Error import invalidInputError
from ipex_llm.transformers.npu_models.lm_head import SlicedLMHead
from ipex_llm.transformers.npu_models.convert import module_optimization
from ipex_llm.transformers.npu_models.linear import QuantizedLinear
from ipex_llm.utils.common.log4Error import invalidInputError


def unpack_auto_round_layer(layer, qtype="sym_int4_rtn"):
    n, m = layer.infeatures, layer.outfeatures
    weight = layer.qweight.to("cpu")
    scale = layer.scales.to("cpu")
    zeros = layer.qzeros.to("cpu")  # np.int32, 1 x m // 4
    bits = layer.bits

    scale = scale.t().contiguous()

    int_weight = torch.zeros((n, m), dtype=torch.uint8)
    num = 32 // bits

    for i in range(0, n // num):
        for j in range(0, num):
            int_weight[i*num + j, :] = ((weight[i, :] >> (j*bits)) & 0x0000000F).to(torch.uint8)

    int_weight = (int_weight - 8).to(torch.int8)  # n, m
    qweights = int_weight.t().contiguous()  # m, n

    # if we want to transform it to our NPU format, uncomment below code
    qweights = qweights.reshape(m, -1, 2)  # m * n/2 * 2
    low_bit, high_bit = qweights.split(1, dim=-1)
    high_bit = high_bit.squeeze().view(torch.int8)
    low_bit = low_bit.squeeze().view(torch.int8)
    high_bit = high_bit << 4
    low_bit = low_bit & 0x0f
    qweights = high_bit | low_bit

    if qtype == "sym_int4_rtn" or qtype == "sym_int8_rtn":
        zero = None
    elif qtype == "asym_int4_rtn":
        zero = zeros.view(torch.int32)
        int_zero = torch.zeros((1, m), dtype=torch.uint8)
        num = 32 // bits

        for i in range(0, m // num):
            for j in range(0, num):
                int_zero[:, i*num + j] = ((zero[:, i] >> (j*bits)) & 0x0000000F).to(torch.uint8)

        zero = int_zero.to(torch.int8)
        zero = zero.t().contiguous()  # m, 1
        zero = zero.to(torch.float32) * -1 * scale
        zero += 8 * scale
    else:
        invalidInputError(False,
                          f"unpack_auto_round_layer does not support qtype {qtype}.")
    return qweights.view(torch.uint8), scale.to(torch.float16), zero.to(torch.float16)


@module_optimization
def replace_with_QuantizedLinear(layer, qtype, device, modules_to_not_convert,
                                 group_size, imatrix):
    from ipex_llm.transformers.low_bit_linear import ggml_convert_qtype
    from ipex_llm.ggml.quantize import ggml_tensor_qtype
    iqtype = ggml_tensor_qtype[qtype]
    if layer.__class__.__name__ == "QuantLinear":
        from auto_round_extension.ipex.qlinear_ipex_gptq import QuantLinear
        if isinstance(layer, QuantLinear):
            # auto-round's QuantLinear
            qweights, scale, zero = unpack_auto_round_layer(layer, qtype=qtype)
            return QuantizedLinear(qweights, scale, zero, layer.bias,
                                   group_size=group_size, qtype=qtype)
    elif isinstance(layer, torch.nn.Linear) and not hasattr(layer, "qtype"):
        enable_scale_search = (os.environ.get("IPEX_LLM_NPU_QUANTIZATION_OPT", "0") != "0" or
                               os.environ.get("IPEX_LLM_NPU_QUANTIZATION_HQQ", "0") != "0")
        qweights, scale = ggml_convert_qtype(layer.weight.data.to(torch.float32),
                                             iqtype, device=device,
                                             enable_scale_search=enable_scale_search,
                                             imatrix=imatrix)
        zero = None
        # split scale to scale & zero
        if qtype == "asym_int4_rtn":
            scale, zero = torch.split(scale, scale.shape[0] // 2)
        return QuantizedLinear(qweights, scale, zero, layer.bias,
                               group_size=group_size, qtype=qtype)


def convert_auto_round_model_to_npu_model(model, save_directory, max_context_len=1024,
                                          max_prompt_len=960, transpose_value_cache=True,
                                          fuse_layers=None, mixed_precision=False,
                                          inter_pp=None, intra_pp=None, optimize_model=True):
    quant_config = getattr(model.config, "quantization_config", None)
    if quant_config is None and quant_config.quant_method != "intel/auto-round":
        exit(-1)

    bits = quant_config.bits
    group_size = quant_config.group_size
    sym = quant_config.sym

    if sym and bits == 4:
        qtype = "sym_int4_rtn"
    elif not sym and bits == 4:
        qtype = "asym_int4_rtn"
    elif sym and bits == 4:
        qtype = "sym_int8_rtn"
    else:
        invalidInputError(False,
                          "Invalid dtype.")

    if group_size == -1:
        quantization_group_size = 0
    else:
        quantization_group_size = group_size

    if model.config.model_type == "qwen2":
        # for Qwen2-7B-Insturct and MiniCPM-V 2.6, divide lm_head into 14 parts
        if model.config.hidden_size == 3584 and (model.config.vocab_size == 152064 or
           model.config.vocab_size == 151666):
            # Do not split lm_head and use sym_int8 instead when mixed_precison is True
            if quantization_group_size == 0:
                # Do not split lm_head and use sym_int8 instead when mixed_precison is True
                is_split = (not mixed_precision) and qtype in ["sym_int4_rtn", "asym_int4_rtn"]
                split_num = 14 if is_split else 1
                new_lm_head = SlicedLMHead(model.lm_head.weight, split_num=split_num,
                                           bias=model.lm_head.bias, use_split=True,
                                           group_size=quantization_group_size,
                                           asym=((qtype == "asym_int4_rtn") and
                                                 (not mixed_precision)))
            del model.lm_head
            model.lm_head = new_lm_head

    replace_with_QuantizedLinear(model, qtype, "cpu", [],
                                 quantization_group_size, None)

    from intel_npu_acceleration_library.compiler import create_npu_kernels
    create_npu_kernels(model)
    model = model.eval()
    model.config.update({"mixed_precision": mixed_precision})
    model.config.update({"group_size": quantization_group_size})
    model.config.update({"asym": qtype == "asym_int4_rtn"})
    model.config.update({"bigdl_transformers_low_bit": qtype})
    model.config.update({"optimize_model": optimize_model})

    if (not hasattr(model, 'llm') and
            model.config.model_type in ["qwen2", "llama", "minicpm"]):
        from ipex_llm.transformers.npu_models.convert import optimize_llm_single_process
        optimize_llm_single_process(
            model,
            kv_len=max_context_len - 1,
            max_prompt_len=max_prompt_len,
            transpose_value_cache=transpose_value_cache,
            group_size=quantization_group_size,
            qtype=qtype,
            save_directory=save_directory,
            fuse_layers=fuse_layers
        )
    else:
        from ipex_llm.transformers.npu_models.convert_mp import optimize_llm
        optimize_llm(
            model,
            max_context_len=max_context_len - 1,
            max_prompt_len=max_prompt_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
            group_size=quantization_group_size
        )

    return model

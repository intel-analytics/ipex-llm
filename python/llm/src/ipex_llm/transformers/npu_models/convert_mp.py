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

import os
import torch
import importlib
import numpy as np
from ipex_llm.transformers.low_bit_linear import LowBitLinear, FP4Params
from ipex_llm.transformers.npu_models.lm_head import SlicedLMHead
from ipex_llm.utils.common.log4Error import invalidInputError


def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)


def optimize_llm_pre(model: torch.nn.Module, qtype, mixed_precision,
                     quantization_group_size=0, load=False, max_prompt_len=512):
    if os.environ.get("IPEX_LLM_NPU_MTL", "0") == "1":
        # For MTL support
        os.environ["IPEX_LLM_NPU_DISABLE_COMPILE_OPT"] = "1"

    if os.environ.get("IPEX_LLM_NPU_ARL", "0") == "1":
        # For ARL support
        os.environ["IPEX_LLM_NPU_DISABLE_COMPILE_OPT"] = "1"

    if model.config.model_type == "baichuan":
        # process NormHead module in Baichuan2 7B
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            vocab_size, hidden_size = model.lm_head.weight.shape
            lm_head_weight_data = model.lm_head.weight.data
            model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False,
                                            device=lm_head_weight_data.device)
            if model.lm_head.weight.data.device != "meta":
                norm_weight = torch.nn.functional.normalize(lm_head_weight_data)
                model.lm_head.weight.data = norm_weight
        if model.config.hidden_size in [4096, 2048]:
            from ipex_llm.transformers.models.baichuan import pre_compute_inv_freq
            model.apply(pre_compute_inv_freq)

    cpu_lm_head = os.environ.get("IPEX_LLM_CPU_LM_HEAD", "0") != "0"

    # workaround for long input performance of llama3.2-3b and glm-edge-4b CW
    if os.environ.get("IPEX_LLM_NPU_DISABLE_COMPILE_OPT") is None:
        disable_compile_opt = model.config.model_type == "llama" and \
            model.config.hidden_size == 3072 and max_prompt_len >= 1920 and \
            quantization_group_size == 0
        os.environ["IPEX_LLM_NPU_DISABLE_COMPILE_OPT"] = "1" if disable_compile_opt else "0"

    # workaround for MiniCPM-2B
    if model.config.model_type == "minicpm" and model.config.num_hidden_layers == 40:
        # 73440 is vocab_size of MiniCPM-1B
        new_linear_0 = torch.nn.Linear(0, 0, bias=False)
        new_weight_0 = torch.nn.Parameter(model.lm_head.weight[:73440, :], requires_grad=False)
        new_linear_0.weight = new_weight_0
        new_linear_0.in_features = new_weight_0.size(1)
        new_linear_0.out_features = new_weight_0.size(0)
        model.lm_head_0 = new_linear_0

        new_linear_1 = torch.nn.Linear(0, 0, bias=False)
        import torch.nn.functional as F
        padded_weight = F.pad(model.lm_head.weight[73440:, :],
                              (0, 0, 0, 73440*2 - model.config.vocab_size))
        new_weight_1 = torch.nn.Parameter(padded_weight, requires_grad=False)
        new_linear_1.weight = new_weight_1
        new_linear_1.in_features = new_weight_1.size(1)
        new_linear_1.out_features = new_weight_1.size(0)
        model.lm_head_1 = new_linear_1
        del model.lm_head

    if model.config.model_type == "minicpmv" and hasattr(model, "llm"):
        # MiniCPM-V
        # convert conv2d and layernorm
        from ipex_llm.transformers.npu_models.minicpmv_mp import MinicpmVPatchEmbedding, \
            replace_with_Layernorm
        origin_conv = model.vpm.embeddings.patch_embedding
        new_conv = MinicpmVPatchEmbedding(
            weight=origin_conv.weight.to(torch.float16),
            bias=origin_conv.bias.to(torch.float16),
            strides=model.config.vision_config.patch_size,
        )
        model.vpm.embeddings.patch_embedding = new_conv
        del new_conv
        replace_with_Layernorm(model, qtype=None, device='NPU',
                               modules_to_not_convert=[], group_size=0)

        # replace forward function
        from ipex_llm.transformers.npu_models.minicpmv_mp import pad_mlp_fc2, pad_mlp_forward, \
            encoder_attn_forward, multi_head_attn_forward, resampler_forward
        model.apply(pad_mlp_fc2)    # pad mlp.fc2 to avoid compile error
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        setattr(module.Resampler, "forward", resampler_forward)
        module = importlib.import_module(modeling_module_name.replace("modeling_minicpmv",
                                                                      "resampler"))
        setattr(module.MultiheadAttention, "multi_head_attention_forward", multi_head_attn_forward)
        if model.config.hidden_size == 3584 and model.config.vocab_size == 151666:
            # MiniCPM-V 2.6
            module = importlib.import_module(modeling_module_name.replace("modeling_minicpmv",
                                                                          "modeling_navit_siglip"))
            setattr(module.SiglipAttention, "forward", encoder_attn_forward)
            setattr(module.SiglipMLP, "forward", pad_mlp_forward)

            # workaround for lm_head on NPU
            from ipex_llm.transformers.npu_models.minicpmv_mp import pad_lm_head, lm_head_forward
            model.apply(pad_lm_head)    # pad lm_head to avoid compile error
            setattr(model.llm.lm_head, "forward", lm_head_forward)
        elif model.config.hidden_size == 4096 and model.config.vocab_size == 128256:
            # MiniCPM-V 2.5
            from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionMLP, \
                Idefics2VisionAttention
            convert_forward(model, Idefics2VisionAttention, encoder_attn_forward)
            convert_forward(model, Idefics2VisionMLP, pad_mlp_forward)

        if model.config.hidden_size == 2304 and model.config.vocab_size == 122753:
            # MiniCPM-V 2
            model.llm.config.model_type = "minicpm"
        elif model.config.hidden_size == 3584 and model.config.vocab_size == 151666:
            # MiniCPM-V 2.6
            model.llm.config.model_type = "qwen2"
        elif model.config.hidden_size == 4096 and model.config.vocab_size == 128256:
            # MiniCPM-V 2.5
            model.llm.config.model_type = "llama"
        model = model.llm

    if model.config.model_type in ["qwen2", "llama", "minicpm", "baichuan"]:
        from ipex_llm.transformers.npu_models.common import split_linears
        if quantization_group_size == 0:
            n_splits_linear = 1
            if qtype in ["sym_int8_rtn", "asym_int4_rtn"]:
                # do not split mlp down_proj for Qwen2-7B & sym_int8
                n_splits_down_proj = 1
            else:
                n_splits_down_proj = 2 if (model.config.intermediate_size == 18944 or
                                           os.environ.get("IPEX_LLM_NPU_MTL", "0") == "1" or
                                           os.environ.get("IPEX_LLM_NPU_ARL", "0") == "1") else 1
        else:
            invalidInputError(
                model.config.hidden_size % quantization_group_size == 0 and
                model.config.intermediate_size % quantization_group_size == 0,
                "The model hidden_size and intermediate_size should be divisible by "
                f"quantization_group_size, but got hidden_size: {model.config.hidden_size}, "
                f"intermediate_size: {model.config.intermediate_size}, and "
                f"quantization_group_size: {quantization_group_size}"
            )
            n_splits_linear = model.config.hidden_size // quantization_group_size
            n_splits_down_proj = model.config.intermediate_size // quantization_group_size
        model.apply(lambda m: split_linears(m, n_splits_hidden_size=n_splits_linear,
                                            n_splits_down_proj=n_splits_down_proj,
                                            load=load))

        if quantization_group_size != 0:
            split_num = model.config.hidden_size // quantization_group_size
            if model.config.model_type == "minicpm" and model.config.num_hidden_layers == 40:
                # workaround for MiniCPM-2B
                new_lm_head_0 = SlicedLMHead(model.lm_head_0.weight, split_num=split_num,
                                             bias=model.lm_head_0.bias, use_split=True,
                                             group_size=quantization_group_size,
                                             asym=(qtype == "asym_int4_rtn"))
                del model.lm_head_0
                model.lm_head_0 = new_lm_head_0
                new_lm_head_1 = SlicedLMHead(model.lm_head_1.weight, split_num=split_num,
                                             bias=model.lm_head_1.bias, use_split=True,
                                             group_size=quantization_group_size,
                                             asym=(qtype == "asym_int4_rtn"))
                del model.lm_head_1
                model.lm_head_1 = new_lm_head_1
            else:
                new_lm_head = SlicedLMHead(model.lm_head.weight, split_num=split_num,
                                           bias=model.lm_head.bias, use_split=True,
                                           group_size=quantization_group_size,
                                           asym=(qtype == "asym_int4_rtn"))
                del model.lm_head
                model.lm_head = new_lm_head

    if model.config.model_type == "qwen2":
        # for Qwen2-7B-Insturct and MiniCPM-V 2.6, divide lm_head into 14 parts
        if model.config.hidden_size == 3584 and (model.config.vocab_size == 152064 or
           model.config.vocab_size == 151666) and not cpu_lm_head:
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

    if model.config.model_type == "xlm-roberta":
        from ipex_llm.transformers.npu_models.xlm_mp import XLMPoolLayer, replace_with_Layernorm
        pooler_dense = model.pooler.dense
        opt_linear = XLMPoolLayer(
            weight=pooler_dense.weight.to(torch.float16),
            bias=pooler_dense.bias.to(torch.float16),
            output_channel=model.config.hidden_size,
            input_channel=model.config.hidden_size
        )
        model.pooler.dense = opt_linear
        replace_with_Layernorm(model.embeddings, qtype=None, device='NPU',
                               modules_to_not_convert=[], group_size=0)

    # lm_head to cpu optimization
    if cpu_lm_head:
        # disable the optimization by default
        from ipex_llm.transformers.low_bit_linear import SYM_INT4, SYM_INT8
        if qtype == "sym_int4_rtn":
            lm_qtype = SYM_INT4
        else:
            lm_qtype = SYM_INT8
        # lm_head opt to mp opt (llama, qwen2)
        optimize_lm_head = model.config.model_type not in ["llama", "qwen2"]
        new_linear = LowBitLinear(model.lm_head.in_features,
                                  model.lm_head.out_features,
                                  lm_qtype,
                                  False,
                                  optimize_lm_head=optimize_lm_head)
        paramsLowBit = FP4Params(data=model.lm_head.weight.data,
                                 requires_grad=False,
                                 quantized=False,
                                 _shape=None,
                                 qtype=lm_qtype,
                                 in_features=model.lm_head.in_features).to("cpu")
        new_linear._parameters['weight'] = paramsLowBit
        model.lm_head = new_linear


def convert_llama(
        model: torch.nn.Module,
        max_output_len=1024,
        max_prompt_len=1024,
        decoder=False,
        inter_pp=None,
        intra_pp=None,
        transpose_value_cache=True,
):
    from ipex_llm.transformers.npu_models.llama_mp import gen_llama_fused_model_forward,\
        gen_llama_32_fused_model_forward
    from ipex_llm.transformers.npu_models.llama_mp import DecodeRunner, PrefillRunner
    from transformers.models.llama.modeling_llama import LlamaModel

    if decoder:
        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
        )
    else:
        decode_runner = None
    prefill_runner = PrefillRunner(
        model,
        max_output_len=max_output_len,
        max_prompt_len=max_prompt_len,
        transpose_value_cache=transpose_value_cache,
    )
    from packaging import version
    import transformers
    trans_version = transformers.__version__
    if version.parse(trans_version) == version.parse("4.45.0"):
        # llama-3.2-3B & llama-3.2-1B
        llama_model_forward = gen_llama_32_fused_model_forward(
            prefill_runner=prefill_runner, decode_runner=decode_runner
        )
    else:
        llama_model_forward = gen_llama_fused_model_forward(
            prefill_runner=prefill_runner, decode_runner=decode_runner
        )
    convert_forward(model, LlamaModel, llama_model_forward)
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from ipex_llm.transformers.npu_models.llama_mp import llama2_casullm_forward
    convert_forward(model, LlamaForCausalLM, llama2_casullm_forward)


def convert_baichuan(
        model: torch.nn.Module,
        max_output_len=1024,
        max_prompt_len=1024,
        decoder=False,
        inter_pp=None,
        intra_pp=None,
        transpose_value_cache=True,
):
    from ipex_llm.transformers.npu_models.baichuan_mp import gen_baichuan_fused_model_forward
    from ipex_llm.transformers.npu_models.baichuan_mp import DecodeRunner, PrefillRunner
    if decoder:
        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
        )
    else:
        decode_runner = None
    prefill_runner = PrefillRunner(
        model,
        max_output_len=max_output_len,
        max_prompt_len=max_prompt_len,
        transpose_value_cache=transpose_value_cache,
    )
    baichuan_model_forward = gen_baichuan_fused_model_forward(
        prefill_runner=prefill_runner, decode_runner=decode_runner
    )
    modeling_module_name = model.__class__.__module__
    module = importlib.import_module(modeling_module_name)
    convert_forward(model, module.BaichuanModel, baichuan_model_forward)
    from ipex_llm.transformers.npu_models.baichuan_mp import baichuan2_causal_forward
    convert_forward(model, module.BaichuanForCausalLM, baichuan2_causal_forward)


def convert_minicpm(
    model: torch.nn.Module,
    max_output_len=1024,
    max_prompt_len=1024,
    decoder=False,
    inter_pp=None,
    intra_pp=None,
    transpose_value_cache=True,
):
    from ipex_llm.transformers.npu_models.minicpm_mp import gen_minicpm_fused_model_forward
    from ipex_llm.transformers.npu_models.minicpm_mp import DecodeRunner, PrefillRunner
    modeling_module_name = model.__class__.__module__
    module = importlib.import_module(modeling_module_name)

    if decoder:
        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
        )
    else:
        decode_runner = None
    prefill_runner = PrefillRunner(
        model,
        max_output_len=max_output_len,
        max_prompt_len=max_prompt_len,
        transpose_value_cache=transpose_value_cache,
    )
    minicpm_model_forward = gen_minicpm_fused_model_forward(
        prefill_runner=prefill_runner, decode_runner=decode_runner
    )
    convert_forward(model, module.MiniCPMModel, minicpm_model_forward)
    if model.config.num_hidden_layers == 40:
        # for minicpm-2b
        from ipex_llm.transformers.npu_models.minicpm_mp import minicpm_casullm_forward
        convert_forward(model, module.MiniCPMForCausalLM, minicpm_casullm_forward)


def convert_qwen(
        model: torch.nn.Module,
        max_output_len=1024,
        max_prompt_len=1024,
        decoder=False,
        inter_pp=None,
        intra_pp=None,
        transpose_value_cache=True,
):
    from ipex_llm.transformers.npu_models.qwen2_mp import gen_qwen2_fused_model_forward
    from ipex_llm.transformers.npu_models.qwen2_mp import DecodeRunner, PrefillRunner
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    if decoder:
        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
        )
    else:
        decode_runner = None
    prefill_runner = PrefillRunner(
        model,
        max_output_len=max_output_len,
        max_prompt_len=max_prompt_len,
        transpose_value_cache=transpose_value_cache,
    )
    qwen2_model_forward = gen_qwen2_fused_model_forward(
        prefill_runner=prefill_runner, decode_runner=decode_runner
    )
    convert_forward(model, Qwen2Model, qwen2_model_forward)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from ipex_llm.transformers.npu_models.qwen2_mp import qwen2_casullm_forward
    convert_forward(model, Qwen2ForCausalLM, qwen2_casullm_forward)


def convert_bce(
    model: torch.nn.Module,
    max_context_len=1024,
    max_prompt_len=1024,
    transpose_value_cache=True,
):
    from ipex_llm.transformers.npu_models.xlm_mp import gen_xlm_fused_encoder_forward
    from ipex_llm.transformers.npu_models.xlm_mp import PrefillRunner
    prefill_runner = PrefillRunner(
        model,
        max_output_len=max_context_len,
        max_prompt_len=max_prompt_len,
        transpose_value_cache=transpose_value_cache,
    )
    encoder_forward = gen_xlm_fused_encoder_forward(
        prefill_runner=prefill_runner
    )
    from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEncoder
    convert_forward(model, XLMRobertaEncoder, encoder_forward)


def optimize_llm(
    model: torch.nn.Module,
    max_context_len=1024,
    max_prompt_len=1024,
    inter_pp=None,
    intra_pp=None,
    transpose_value_cache=True,
    group_size=0
):
    if model.config.model_type == "llama":
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            if group_size == 0:
                inter_pp = 2
            # llama3.2
            elif model.config.intermediate_size == 8192:
                # llama3.2 1b
                if model.config.hidden_size == 2048:
                    inter_pp = 1
                else:
                    inter_pp = 2
            else:
                inter_pp = 8
        convert_llama(model,
                      max_output_len=max_context_len,
                      max_prompt_len=max_prompt_len,
                      inter_pp=inter_pp,
                      intra_pp=intra_pp,
                      decoder=True,
                      transpose_value_cache=transpose_value_cache)
    elif model.config.model_type == "qwen2":
        # for qwen2-1.5B, qwen2-7B, qwen2.5-3B
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            if model.config.intermediate_size == 18944:
                if group_size != 0:
                    inter_pp = 5
                else:
                    inter_pp = 2
            else:
                inter_pp = 1
        convert_qwen(model,
                     max_output_len=max_context_len,
                     max_prompt_len=max_prompt_len,
                     inter_pp=inter_pp,
                     intra_pp=intra_pp,
                     decoder=True,
                     transpose_value_cache=transpose_value_cache)
    elif model.config.model_type == "minicpm":
        # for minicpm-1b
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 2
        convert_minicpm(model,
                        max_output_len=max_context_len,
                        max_prompt_len=max_prompt_len,
                        inter_pp=inter_pp,
                        intra_pp=intra_pp,
                        decoder=True,
                        transpose_value_cache=transpose_value_cache)
    elif model.config.model_type == "baichuan" and model.config.num_hidden_layers == 32:
        # for Baichuan2-7B
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 2 if group_size == 0 else 4
        convert_baichuan(model,
                         max_output_len=max_context_len,
                         max_prompt_len=max_prompt_len,
                         inter_pp=inter_pp,
                         intra_pp=intra_pp,
                         decoder=True,
                         transpose_value_cache=transpose_value_cache)
    elif model.config.model_type == "xlm-roberta":
        # for bce-embedding-base_v1
        convert_bce(model,
                    max_context_len=max_context_len,
                    max_prompt_len=max_prompt_len,
                    transpose_value_cache=transpose_value_cache)
    if hasattr(model, 'lm_head') and isinstance(model.lm_head, SlicedLMHead):
        model.lm_head.get_fused_lm_head()
    # MiniCPM-2b
    if hasattr(model, "lm_head_1") and isinstance(model.lm_head_1, SlicedLMHead):
        model.lm_head_1.get_fused_lm_head()
        model.lm_head_0.get_fused_lm_head()


def optimize_funasr(
    model: torch.nn.Module,
    max_context_len=1024,
    max_prompt_len=1024,
    inter_pp=None,
    intra_pp=None,
    transpose_value_cache=True,
):
    if intra_pp is None:
        intra_pp = 2
    if inter_pp is None:
        inter_pp = 2
    from ipex_llm.transformers.npu_models.paraformer_mp import gen_funasr_fused_encoder_forward, \
        gen_funasr_fused_decoder_forward
    from ipex_llm.transformers.npu_models.paraformer_mp import PrefillRunner, DecodeRunner
    prefill_runner = PrefillRunner(
        model,
        max_output_len=max_context_len,
        max_prompt_len=max_prompt_len,
        transpose_value_cache=transpose_value_cache,
    )
    encoder_forward = gen_funasr_fused_encoder_forward(
        prefill_runner=prefill_runner
    )
    decode_runner = DecodeRunner(
        model,
        max_seq_len=max_context_len,
        inter_pp=inter_pp,
        intra_pp=intra_pp,
        transpose_value_cache=transpose_value_cache,
    )
    decoder_forward = gen_funasr_fused_decoder_forward(
        decode_runner=decode_runner
    )
    from funasr.models.sanm.encoder import SANMEncoder
    from funasr.models.paraformer.decoder import ParaformerSANMDecoder
    convert_forward(model.model, SANMEncoder, encoder_forward)
    convert_forward(model.model, ParaformerSANMDecoder, decoder_forward)

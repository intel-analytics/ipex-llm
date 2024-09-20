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
from ipex_llm.transformers.npu_models.lm_head import LMHeadLinear, SlicedLMHead


def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)


def optimize_llm_pre(model: torch.nn.Module, qtype, mixed_precision):
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

    # MiniCPM-V 2.6 must put lm_head on CPU now
    cpu_lm_head = (
        (model.config.model_type == "minicpmv" and model.config.hidden_size == 3584 and
         model.config.vocab_size == 151666)
        or os.environ.get("IPEX_LLM_CPU_LM_HEAD", "0") != "0"
    )

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

    if model.config.model_type == "qwen2":
        from ipex_llm.transformers.npu_models.qwen2_mp import split_mlp_down_proj
        model.apply(split_mlp_down_proj)

        # for Qwen2-7B-Insturct, divide lm_head into 14 parts
        if model.config.hidden_size == 3584 and model.config.vocab_size == 152064 and \
                not cpu_lm_head:
            # Do not split lm_head and use sym_int8 instead when mixed_precison is True
            is_split = (not mixed_precision) and qtype == "sym_int4_rtn"
            split_num = 14 if is_split else 1
            new_lm_head = SlicedLMHead(model.lm_head.weight, split_num=split_num,
                                       bias=model.lm_head.bias)
            del model.lm_head
            model.lm_head = new_lm_head

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


def optimize_llm(
    model: torch.nn.Module,
    max_output_len=1024,
    max_prompt_len=1024,
    inter_pp=None,
    intra_pp=None,
    transpose_value_cache=True,
):
    if model.config.model_type == "llama":
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 2

        from ipex_llm.transformers.npu_models.llama_mp import gen_llama_fused_model_forward
        from ipex_llm.transformers.npu_models.llama_mp import DecodeRunner, PrefillRunner
        from transformers.models.llama.modeling_llama import LlamaModel

        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
        )
        prefill_runner = PrefillRunner(
            model,
            max_output_len=max_output_len,
            max_prompt_len=max_prompt_len,
            transpose_value_cache=transpose_value_cache,
        )
        llama_model_forward = gen_llama_fused_model_forward(
            prefill_runner=prefill_runner, decode_runner=decode_runner
        )
        convert_forward(model, LlamaModel, llama_model_forward)
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from ipex_llm.transformers.npu_models.llama_mp import llama2_casullm_forward
        convert_forward(model, LlamaForCausalLM, llama2_casullm_forward)
    elif model.config.model_type == "qwen2" and model.config.num_hidden_layers == 28:
        # for qwen2-1.5B and qwen2-7B
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 2 if model.config.intermediate_size == 18944 else 1

        from ipex_llm.transformers.npu_models.qwen2_mp import gen_qwen2_fused_model_forward
        from ipex_llm.transformers.npu_models.qwen2_mp import DecodeRunner, PrefillRunner
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
        )
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

        # for Qwen2-7B-Insturct, divide lm_head into 14 parts
        if model.config.hidden_size == 3584 and model.config.vocab_size == 152064 and \
                isinstance(model.lm_head, SlicedLMHead):
            model.lm_head.get_fused_lm_head()
    elif model.config.model_type == "minicpm":
        # for minicpm-1b
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 2

        from ipex_llm.transformers.npu_models.minicpm_mp import gen_minicpm_fused_model_forward
        from ipex_llm.transformers.npu_models.minicpm_mp import DecodeRunner, PrefillRunner

        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)

        if model.config.num_hidden_layers == 52:
            # for minicpm-1b
            transpose_cache = transpose_value_cache
        elif model.config.num_hidden_layers == 40:
            # for minicpm-2b
            transpose_cache = False

        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_cache,
        )
        prefill_runner = PrefillRunner(
            model,
            max_output_len=max_output_len,
            max_prompt_len=max_prompt_len,
            transpose_value_cache=transpose_cache,
        )
        minicpm_model_forward = gen_minicpm_fused_model_forward(
            prefill_runner=prefill_runner, decode_runner=decode_runner
        )
        convert_forward(model, module.MiniCPMModel, minicpm_model_forward)
        if model.config.num_hidden_layers == 40:
            # for minicpm-2b
            from ipex_llm.transformers.npu_models.minicpm_mp import minicpm_casullm_forward
            convert_forward(model, module.MiniCPMForCausalLM, minicpm_casullm_forward)
    elif model.config.model_type == "baichuan" and model.config.num_hidden_layers == 32:
        # for Baichuan2-7B
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 2
        from ipex_llm.transformers.npu_models.baichuan_mp import gen_baichuan_fused_model_forward
        from ipex_llm.transformers.npu_models.baichuan_mp import DecodeRunner, PrefillRunner
        decode_runner = DecodeRunner(
            model,
            max_seq_len=max_output_len,
            inter_pp=inter_pp,
            intra_pp=intra_pp,
            transpose_value_cache=transpose_value_cache,
        )
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

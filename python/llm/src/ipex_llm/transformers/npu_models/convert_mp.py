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
from ipex_llm.transformers.low_bit_linear import LowBitLinear, FP4Params


def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)


def optimize_llm_pre(model: torch.nn.Module, qtype):
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

    # lm_head to cpu optimization
    if os.environ.get("IPEX_LLM_CPU_LM_HEAD", "1") != "0":
        is_unsupported_model = (model.config.model_type == "llama"
                                and model.vocab_size > 32000)
        if not is_unsupported_model:
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
    elif model.config.model_type == "qwen2" and model.config.intermediate_size == 8960:
        # for qwen2-1.5B
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 1

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
    elif model.config.model_type == "stablelm" and model.config.num_hidden_layers == 32:
        # for stablelm-zephyr-3b
        if intra_pp is None:
            intra_pp = 2
        if inter_pp is None:
            inter_pp = 2
        from ipex_llm.transformers.npu_models.stablelm_mp import gen_stablelm_fused_model_forward
        from ipex_llm.transformers.npu_models.stablelm_mp import DecodeRunner, PrefillRunner
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
        stablelm_model_forward = gen_stablelm_fused_model_forward(
            prefill_runner=prefill_runner, decode_runner=decode_runner
        )
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        convert_forward(model, module.StableLmModel, stablelm_model_forward)

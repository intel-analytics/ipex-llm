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


def convert_forward(m, target_m, new_forward):
    if m.__class__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
    for _, sub_m in m.named_children():
        convert_forward(sub_m, target_m, new_forward)


def optimize_llm(
    model: torch.nn.Module,
    max_output_len=1024,
    max_prompt_len=1024,
    inter_pp=2,
    intra_pp=2,
    transpose_value_cache=True,
):
    if model.config.model_type == "llama":
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

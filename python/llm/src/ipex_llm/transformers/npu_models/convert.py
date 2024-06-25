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


def optimize_llm(model: torch.nn.Module):
    if model.config.model_type == "llama":
        from ipex_llm.transformers.npu_models.llama import merge_qkv
        model.apply(merge_qkv)
        from ipex_llm.transformers.npu_models.llama import llama_attention_forward
        from transformers.models.llama.modeling_llama import LlamaAttention
        convert_forward(model, LlamaAttention, llama_attention_forward)

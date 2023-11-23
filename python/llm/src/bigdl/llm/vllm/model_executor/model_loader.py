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
# https://github.com/vllm-project/vllm/blob/v0.2.1.post1/vllm/model_executor/model_loader.py
# which is licensed under Apache License 2.0
#
# Copyright 2023 The vLLM team. All rights reserved.
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
"""Utilities for selecting and loading models."""
import contextlib
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from bigdl.llm.vllm.config import ModelConfig
from bigdl.llm.vllm.model_executor.models.bigdl_llama import BigDLLlamaForCausalLM
from bigdl.llm.utils.common import invalidInputError

# bigdl-llm Intel specified code change
# bigdl-llm change start
# summary: Currently we only support LLAMA model and users can add their models by adding
# code in ./models dir and then regitering here.

_MODEL_REGISTRY = {
    # "AquilaModel": AquilaForCausalLM,
    # "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    # "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    # "BloomForCausalLM": BloomForCausalLM,
    # "FalconForCausalLM": FalconForCausalLM,
    # "GPT2LMHeadModel": GPT2LMHeadModel,
    # "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    # "GPTJForCausalLM": GPTJForCausalLM,
    # "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    # "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": BigDLLlamaForCausalLM,
    # "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    # "MistralForCausalLM": MistralForCausalLM,
    # "MPTForCausalLM": MPTForCausalLM,
    # "OPTForCausalLM": OPTForCausalLM,
    # "QWenLMHeadModel": QWenLMHeadModel,
    # "RWForCausalLM": FalconForCausalLM,
}

_MODEL_CLASSES_SUPPORT_QUANTIZATION = [
    #     LlamaForCausalLM,
]

# bigdl-llm change end


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    invalidInputError(
        False,
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the quantization config.
    quant_config = None
    if model_config.quantization is not None:
        invalidInputError(f"Quantization is not supported for {model_class}.")

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        if model_class in _MODEL_CLASSES_SUPPORT_QUANTIZATION:
            model = model_class(model_config.hf_config, quant_config)
        else:
            model = model_class(model_config.hf_config, device=model_config.device,
                                max_model_len=model_config.max_model_len)
        # Load the weights from the cached or downloaded files.
        model.load_weights(model_config.model, model_config.download_dir,
                           model_config.load_format, model_config.revision)
        # bigdl-llm Intel specified code change
        # bigdl-llm change start
        # summary: Only use cuda when device is gpu.
        if model_config.device == 'gpu':
            model = model.cuda()
        # bigdl-llm change end
    return model.eval()

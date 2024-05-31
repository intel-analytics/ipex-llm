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

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

"""Wrappers on top of large language models APIs."""
from typing import Dict, Type
from langchain.llms.base import BaseLLM

from .bigdlllm import *
from .transformersllm import TransformersLLM
from .transformerspipelinellm import TransformersPipelineLLM

__all__ = [
    "BigdlNativeLLM",
    "LlamaLLM",
    "BloomLLM",
    "GptneoxLLM",
    "StarcoderLLM",
    "TransformersLLM",
    "TransformersPipelineLLM"
]

type_to_cls_dict: Dict[str, Type[BaseLLM]] = {
    "BigdlNativeLLM": BigdlNativeLLM,
    "LlamaLLM": LlamaLLM,
    "BloomLLM": BloomLLM,
    "GptneoxLLM": GptneoxLLM,
    "StarcoderLLM": StarcoderLLM,
    "TransformersPipelineLLM": TransformersPipelineLLM,
    "TransformersLLM": TransformersLLM
}

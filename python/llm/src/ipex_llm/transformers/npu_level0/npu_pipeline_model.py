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

import os
import copy
import types
import warnings
import torch
import transformers
from typing import List
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from . import pipeline_cpp
from .pipeline_cpp import InitLLMPipeline, generate_serve


def patch_flash_attn_import(filename: str) -> List[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def ignore_argument(kwargs: dict, key: "str"):
    arg = kwargs.pop(key, None)
    if arg is not None:
        warnings.warn(f"argument `{key}={arg}` will be ignored")


class _BaseAutoModelClass:
    HF_MODEL = None

    @classmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    def from_pretrained(cls, *args, **kwargs):
        """
        Load a model from a directory or the HF Hub. 
        Not supported for NPU Level Zero Model now.
        """
        pass

    @classmethod
    def load_from_blob(cls, kv_len: int, num_head: int, head_dim: int, num_layers: int, vocab_size: int,
                       model_weight_dir: str, model_name: str,
                       first_blob_name: str, last_blob_name: str, rest_blob_name: str):
        InitLLMPipeline(kv_len, num_head, head_dim, num_layers, vocab_size,
                        model_weight_dir, model_name,
                        first_blob_name, last_blob_name, rest_blob_name)
        cls.kv_len = kv_len
        cls.num_head = num_head
        cls.head_dim = head_dim
        cls.num_layers = num_layers

    @classmethod
    def generate():
        pass


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSpeechSeq2Seq


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSeq2SeqLM


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForSequenceClassification


class AutoModelForMaskedLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForMaskedLM


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForQuestionAnswering


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForNextSentencePrediction


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForMultipleChoice


class AutoModelForTokenClassification(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForTokenClassification

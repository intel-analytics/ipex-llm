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
import types
import warnings
import torch
import transformers
from typing import List
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

import intel_npu_acceleration_library as npu_lib

from ipex_llm.utils.common.log4Error import invalidInputError
from ipex_llm.transformers.utils import logger


def patch_flash_attn_import(filename: str) -> List[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def ignore_argument(kwargs: dict, key: 'str'):
    arg = kwargs.pop(key, None)
    if arg is not None:
        warnings.warn(f"argument `{key}={arg}` will be ignored")


class _BaseAutoModelClass:
    HF_MODEL = None

    @classmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    def from_pretrained(cls,
                        *args,
                        **kwargs):
        """
        Load a model from a directory or the HF Hub. Use load_in_low_bit parameter to convert
        model to low-bit format, like int4 and int8.
        The loaded model will run supported OPs on NPU, then run other OPs on CPU.

        Three new arguments are added to extend Hugging Face's from_pretrained method as follows:
        :param load_in_low_bit: str value, options are ``'sym_int4'``, ``'sym_int8'``,
                                ``'fp16'``, ``'fp32'``.
                                Relevant low bit optimizations will be applied to the model.
        :return: a model instance
        """
        if kwargs.get('device_map', None) not in [None, 'cpu', 'auto']:
            warnings.warn("`device_map` will be ignored")
        kwargs['device_map'] = 'cpu'

        if kwargs.get('torch_dtype', None) not in [None, 'auto', torch.float]:
            warnings.warn("`torch_dtype` will be ignored, `torch.float` will be used")
        kwargs['torch_dtype'] = torch.float

        low_bit = kwargs.pop('load_in_low_bit', 'fp32')
        try:
            # for intel_npu_acceleration_library >= 1.1.0
            from intel_npu_acceleration_library.dtypes import int8, int4
            qtype_map = {
                'sym_int4': int4,
                'sym_int8': int8,
                'fp16': torch.half,
                'fp32': torch.float,
            }
        except ImportError as _e:
            # for intel_npu_acceleration_library < 1.1.0
            qtype_map = {
                'sym_int8': torch.int8,
                'fp16': torch.half,
                'fp32': torch.float,
            }
        invalidInputError(low_bit in qtype_map.keys(),
                          f"unsupported low_bit: {low_bit}, "
                          f"only {list(qtype_map.keys())} are supported")
        qtype = qtype_map[low_bit]

        kwargs["low_cpu_mem_usage"] = True

        # ignore following arguments
        ignore_argument(kwargs, "model_hub")
        ignore_argument(kwargs, "lightweight_bmm")
        ignore_argument(kwargs, "load_in_4bit")
        ignore_argument(kwargs, "load_in_8bit")
        ignore_argument(kwargs, "imatrix")
        ignore_argument(kwargs, "mixed_precision")
        ignore_argument(kwargs, "cpu_embedding")
        ignore_argument(kwargs, "embedding_qtype")
        ignore_argument(kwargs, "optimize_model")
        ignore_argument(kwargs, "modules_to_not_convert")
        ignore_argument(kwargs, "quantization_config")
        ignore_argument(kwargs, "speculative")
        ignore_argument(kwargs, "pipeline_parallel_stages")

        model = cls.HF_Model.from_pretrained(*args, **kwargs)

        logger.info(f"Converting model, it may takes up to several minutes ...")
        model = npu_lib.compile(model, qtype, False)

        # add save_low_bit to pretrained model dynamically
        model.save_low_bit = types.MethodType(cls.save_low_bit, model)

        return model

    @staticmethod
    def save_low_bit(self, model_dir: str, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        model_name = "pytorch_npu_model.pt"
        model_path = os.path.join(model_dir, model_name)
        del self.save_low_bit   # workaround a bug
        torch.save(self, model_path)

    @staticmethod
    @patch("transformers.dynamic_module_utils.get_imports", patch_flash_attn_import)
    def load_low_bit(model_dir: str, *args, **kwargs):
        if kwargs.pop('torch_dtype', None) not in [None, 'auto', torch.float]:
            warnings.warn("`torch_dtype` will be ignored, `torch.float` will be used")

        # ignore following arguments
        ignore_argument(kwargs, "model_hub")
        ignore_argument(kwargs, "lightweight_bmm")
        ignore_argument(kwargs, "cpu_embedding")
        ignore_argument(kwargs, "embedding_qtype")
        ignore_argument(kwargs, "optimize_model")
        ignore_argument(kwargs, "modules_to_not_convert")
        ignore_argument(kwargs, "speculative")
        ignore_argument(kwargs, "pipeline_parallel_stages")

        model_name = "pytorch_npu_model.pt"
        model_path = os.path.join(model_dir, model_name)
        return torch.load(model_path)


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

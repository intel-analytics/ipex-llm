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

import importlib
import logging

from ipex_llm.utils.common import invalidInputError
from .model import *


class BigdlNativeForCausalLM:
    """
    A generic model class that mimics the behavior of
    ``transformers.LlamaForCausalLM.from_pretrained`` API
    """

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        model_family: str = 'llama',
                        dtype: str = 'int4',
                        **kwargs):
        """
        :param pretrained_model_name_or_path: Path for converted BigDL-LLM optimized ggml
               binary checkpoint. The checkpoint should be converted by ``ipex_llm.llm_convert``.
        :param model_family: The model family of the pretrained checkpoint.
               Currently we support ``"llama"``, ``"bloom"``, ``"gptneox"``, ``"starcoder"``.
        :param dtype: Which quantized precision will be converted.
                Now only `int4` and `int8` are supported, and `int8` only works for `llama`
                , `gptneox` and `starcoder`.
        :param cache_dir: (optional) This parameter will only be used when
               ``pretrained_model_name_or_path`` is a huggingface checkpoint or hub repo id.
               It indicates the saving path for the converted low precision model.
        :param tmp_path: (optional) Which path to store the intermediate fp16 model during the
               conversion process. Default to `None` so that intermediate model will not be saved.
        :param kwargs: keyword arguments which will be passed to the model instance

        :return: a model instance
        """
        logging.warning("BigdlNativeForCausalLM has been deprecated, "
                        "please switch to the new CausalLM API for sepcific models.")
        invalidInputError(model_family in ['llama', 'gptneox', 'bloom', 'starcoder'],
                          "Now we only support model family: 'llama', 'gptneox', 'bloom',"
                          " 'starcoder', '{}' is not in the list.".format(model_family))
        invalidInputError(dtype.lower() in ['int4', 'int8'],
                          "Now we only support int4 and int8 as date type for weight")

        ggml_model_path = pretrained_model_name_or_path

        if model_family == 'llama':
            from ipex_llm.ggml.model.llama import Llama
            return Llama(model_path=ggml_model_path, **kwargs)
        elif model_family == 'gptneox':
            from ipex_llm.ggml.model.gptneox import Gptneox
            return Gptneox(model_path=ggml_model_path, **kwargs)
        elif model_family == 'bloom':
            from ipex_llm.ggml.model.bloom import Bloom
            return Bloom(model_path=ggml_model_path, **kwargs)
        elif model_family == 'starcoder':
            from ipex_llm.ggml.model.starcoder import Starcoder
            return Starcoder(model_path=ggml_model_path, **kwargs)


class _BaseGGMLClass:

    GGML_Model = None
    HF_Class = None

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        native: bool = True,
                        dtype: str = "int4",
                        *args,
                        **kwargs):
        """
        :param pretrained_model_name_or_path: Path for model checkpoint.
               If running with ``native int4``, the path should be converted BigDL-LLM optimized
               ggml binary checkpoint, which should be converted by ``ipex_llm.llm_convert``.
               If running with ``transformers int4``, the path should be the huggingface repo id
               to be downloaded or the huggingface checkpoint folder.
        :param native: Load model to either BigDL-LLM optimized Transformer or Native (ggml) int4.
        :param dtype: Which quantized precision will be converted.
               Now only `int4` and `int8` are supported, and `int8` only works for `llama`
               , `gptneox` and `starcoder`.
        :param kwargs: keyword arguments which will be passed to the model instance.

        :return: a model instance
        """
        try:
            if native:
                module = importlib.import_module(cls.GGML_Module)
                class_ = getattr(module, cls.GGML_Model)
                invalidInputError(dtype.lower() in ['int4', 'int8'],
                                  "Now we only support int4 and int8 as date type for weight")
                ggml_model_path = pretrained_model_name_or_path
                model = class_(model_path=ggml_model_path, **kwargs)
            else:
                model = cls.HF_Class.from_pretrained(pretrained_model_name_or_path,
                                                     *args, **kwargs)
        except Exception as e:
            invalidInputError(
                False,
                f"Could not load model from path: {pretrained_model_name_or_path}. "
                f"Please make sure the CausalLM class matches "
                "the model you want to load."
                f"Received error {e}"
            )
        return model


class LlamaForCausalLM(_BaseGGMLClass):
    GGML_Module = "ipex_llm.models"
    GGML_Model = "Llama"
    HF_Class = AutoModelForCausalLM


class ChatGLMForCausalLM(_BaseGGMLClass):
    GGML_Module = "ipex_llm.ggml.model.chatglm"
    GGML_Model = "ChatGLM"
    HF_Class = AutoModel


class GptneoxForCausalLM(_BaseGGMLClass):
    GGML_Module = "ipex_llm.models"
    GGML_Model = "Gptneox"
    HF_Class = AutoModelForCausalLM


class BloomForCausalLM(_BaseGGMLClass):
    GGML_Module = "ipex_llm.models"
    GGML_Model = "Bloom"
    HF_Class = AutoModelForCausalLM


class StarcoderForCausalLM(_BaseGGMLClass):
    GGML_Module = "ipex_llm.models"
    GGML_Model = "Starcoder"
    HF_Class = AutoModelForCausalLM

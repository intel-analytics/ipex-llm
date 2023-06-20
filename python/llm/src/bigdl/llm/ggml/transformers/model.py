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

import os
import traceback
from bigdl.llm.utils.common import invalidInputError


class AutoModelForCausalLM:
    """
    A generic model class that mimics the behavior of
    ``transformers.AutoModelForCausalLM.from_pretrained`` API
    """

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        model_format: str = 'pth',
                        model_family: str = 'llama',
                        dtype: str = 'int4',
                        cache_dir: str = './',
                        tmp_path: str = None,
                        **kwargs):
        """
        :param pretrained_model_name_or_path: We support 3 kinds of pretrained model checkpoint

               1. Path to directory for Hugging Face checkpoint that are directly pulled from
                  Hugging Face hub.

                  If ``model_format='pth'``, the folder should contain: weight bin, tokenizer
                  config, tokenizer.model (required for llama) and added_tokens.json (if applied).
                  For lora fine tuned model, the path should be pointed to a merged weight.

                  If ``model_format='gptq'``, the folder should be be a Hugging Face checkpoint
                  in GPTQ format, which contains weights in pytorch's .pt format,
                  and ``tokenizer.model``.

               2. Path for converted BigDL-LLM optimized ggml binary checkpoint.
                  The checkpoint should be converted by ``bigdl.llm.llm_convert``.
               3. A str for Hugging Face hub repo id.

        :param model_format: Specify the model format to be converted. ``pth`` is for
               PyTorch model checkpoint from Hugging Face. ``gptq`` is for GPTQ format
               model from Hugging Face.
        :param model_family: The model family of the pretrained checkpoint.
               Currently we support ``"llama"``, ``"bloom"``, ``"gptneox"`` and ``"starcoder"``.
        :param dtype: Which quantized precision will be converted.
                Now only `int4` and `int8` are supported, and `int8` only works for `llama`
                , `gptneox` and `starcoder`.
        :param cache_dir: (optional) This parameter will only be used when
               ``pretrained_model_name_or_path`` is a hugginface checkpoint or hub repo id.
               It indicates the saving path for the converted low precision model.
        :param tmp_path: (optional) Which path to store the intermediate fp16 model during the
               conversion process. Default to `None` so that intermediate model will not be saved.
        :param **kwargs: keyword arguments which will be passed to the model instance

        :return: a model instance
        """
        invalidInputError(model_family in ['llama', 'gptneox', 'bloom', 'starcoder'],
                          "Now we only support model family: 'llama', 'gptneox', 'bloom',"
                          " 'starcoder', '{}' is not in the list.".format(model_family))
        invalidInputError(dtype.lower() in ['int4', 'int8'],
                          "Now we only support int4 and int8 as date type for weight")

        # check whether pretrained_model_name_or_path exists.
        # if not, it is likely that the user wants to pass in the repo id.
        if not os.path.exists(pretrained_model_name_or_path):
            try:
                # download from Hugging Face based on repo id
                from huggingface_hub import snapshot_download
                pretrained_model_name_or_path = snapshot_download(
                    repo_id=pretrained_model_name_or_path)
            except Exception as e:
                traceback.print_exc()
                # if downloading fails, it could be the case that repo id is invalid,
                # or the user pass in the wrong path for checkpoint
                invalidInputError(False,
                                  "Downloadng from Hugging Face repo id {} failed. "
                                  "Please input valid Hugging Face hub repo id, "
                                  "or provide the valid path to Hugging Face / "
                                  "BigDL-LLM optimized ggml binary checkpoint, "
                                  "for pretrained_model_name_or_path"
                                  .format(pretrained_model_name_or_path))

        ggml_model_path = pretrained_model_name_or_path
        # check whether pretrained_model_name_or_path is a file.
        # if not, it is likely that pretrained_model_name_or_path
        # points to a Hugging Face checkpoint
        if not os.path.isfile(pretrained_model_name_or_path):
            # Hugging Face checkpoint
            from bigdl.llm import llm_convert
            ggml_model_path = llm_convert(model=pretrained_model_name_or_path,
                                          outfile=cache_dir,
                                          model_family=model_family,
                                          outtype=dtype,
                                          model_format=model_format,
                                          tmp_path=tmp_path)

        if model_family == 'llama':
            from bigdl.llm.ggml.model.llama import Llama
            return Llama(model_path=ggml_model_path, **kwargs)
        elif model_family == 'gptneox':
            from bigdl.llm.ggml.model.gptneox import Gptneox
            return Gptneox(model_path=ggml_model_path, **kwargs)
        elif model_family == 'bloom':
            from bigdl.llm.ggml.model.bloom import Bloom
            return Bloom(model_path=ggml_model_path, **kwargs)
        elif model_family == 'starcoder':
            from bigdl.llm.ggml.model.starcoder import Starcoder
            return Starcoder(model_path=ggml_model_path, **kwargs)

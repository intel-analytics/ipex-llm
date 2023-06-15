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
                        model_family: str = 'llama',
                        dtype: str = 'int4',
                        cache_dir: str = './',
                        **kwargs):
        """
        :param pretrained_model_name_or_path: We support 3 kinds of pretrained model checkpoint

               1. path for huggingface checkpoint that are directly pulled from huggingface hub.
                  This should be a dir path that contains: weight bin, tokenizer config,
                  tokenizer.model (required for llama) and added_tokens.json (if applied).
                  For lora fine tuned model, the path should be pointed to a merged weight.
               2. path for converted ggml binary checkpoint. The checkpoint should be converted by
                  ``bigdl.llm.ggml.convert_model``.
               3. a str for huggingface hub repo id.

        :param model_family: the model family of the pretrained checkpoint.
               Currently we support ``"llama"``, ``"bloom"``, ``"gptneox"``.
        :param dtype: (optional) the data type for weight. Currently we only support ``"int4"``
        :param cache_dir: (optional) this parameter will only be used when
               ``pretrained_model_name_or_path`` is a hugginface checkpoint or hub repo id.
               It indicates the saving path for the converted low precision model.
        :param **kwargs: keyword arguments which will be passed to the model instance

        :return: a model instance
        """
        invalidInputError(model_family in ['llama', 'gptneox', 'bloom'],
                          "Now we only support model family: 'llama', 'gptneox', 'bloom', "
                          "'{}' is not in the list.".format(model_family))
        invalidInputError(dtype.lower() == 'int4',
                          "Now we only support int4 as date type for weight")

        # check whether pretrained_model_name_or_path exists.
        # if not, it is likely that the user wants to pass in the repo id.
        if not os.path.exists(pretrained_model_name_or_path):
            try:
                # download from huggingface based on repo id
                from huggingface_hub import snapshot_download
                pretrained_model_name_or_path = snapshot_download(
                    repo_id=pretrained_model_name_or_path)
            except Exception as e:
                traceback.print_exc()
                # if downloading fails, it could be the case that repo id is invalid,
                # or the user pass in the wrong path for checkpoint
                invalidInputError(False,
                                  "Downloadng from huggingface repo id {} failed. "
                                  "Please input valid huggingface hub repo id, "
                                  "or provide the valid path to huggingface / "
                                  "ggml binary checkpoint, for pretrained_model_name_or_path"
                                  .format(pretrained_model_name_or_path))

        ggml_model_path = pretrained_model_name_or_path
        # check whether pretrained_model_name_or_path is a file.
        # if not, it is likely that pretrained_model_name_or_path
        # points to a huggingface checkpoint
        if not os.path.isfile(pretrained_model_name_or_path):
            # huggingface checkpoint
            from bigdl.llm.ggml import convert_model
            ggml_model_path = convert_model(input_path=pretrained_model_name_or_path,
                                            output_path=cache_dir,
                                            model_family=model_family,
                                            dtype=dtype)

        if model_family == 'llama':
            from bigdl.llm.ggml.model.llama import Llama
            return Llama(model_path=ggml_model_path, **kwargs)
        elif model_family == 'gptneox':
            from bigdl.llm.ggml.model.gptneox import Gptneox
            return Gptneox(model_path=ggml_model_path, **kwargs)
        elif model_family == 'bloom':
            from bigdl.llm.ggml.model.bloom import Bloom
            return Bloom(model_path=ggml_model_path, **kwargs)

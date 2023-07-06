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

from bigdl.llm.utils.common import invalidInputError


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
               binary checkpoint. The checkpoint should be converted by ``bigdl.llm.llm_convert``.
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

        ggml_model_path = pretrained_model_name_or_path

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

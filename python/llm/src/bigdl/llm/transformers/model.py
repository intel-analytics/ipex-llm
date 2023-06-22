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

import transformers
import torch


class _BaseAutoModelClass:

    HF_MODEL = None

    @classmethod
    def from_pretrained(cls,
                        *args,
                        **kwargs):
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        model = cls.HF_Model.from_pretrained(*args, **kwargs)

        if load_in_4bit:
            from .convert import ggml_convert_int4
            model = model.to("cpu", torch.float32)
            model = ggml_convert_int4(model)

        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel

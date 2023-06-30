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
        load_in_4bit = kwargs.get("load_in_4bit", False)
        # if load_in_4bit:
        #     kwargs["low_cpu_mem_usage"] = True
        if load_in_4bit:
            from bigdl.llm.transformers.linear_int4 import LinearInt4, ParamsInt4
            import bitsandbytes as bnb
            old_linear = bnb.nn.Linear4bit
            old_param = bnb.nn.Params4bit
            bnb.nn.Linear4bit = LinearInt4
            bnb.nn.Params4bit = ParamsInt4
            if "torch_dtype" not in kwargs:
                kwargs["torch_dtype"] = torch.float32
            kwargs["device_map"] = "cpu"
            # workaround HF transformers issue
            kwargs["llm_int8_skip_modules"] = []
            kwargs["tie_word_embeddings"] = False

            try:
                model = cls.HF_Model.from_pretrained(*args, **kwargs)
            finally:
                bnb.nn.Linear4bit = old_linear
                bnb.nn.Params4bit = old_param
        return model


class AutoModelForCausalLM(_BaseAutoModelClass):
    HF_Model = transformers.AutoModelForCausalLM


class AutoModel(_BaseAutoModelClass):
    HF_Model = transformers.AutoModel

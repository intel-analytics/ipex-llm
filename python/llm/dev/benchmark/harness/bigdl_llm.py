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

from bigdl.llm.transformers import AutoModelForCausalLM

import inspect
from lm_eval.models.huggingface import AutoCausalLM
from functools import partial

class BigDLLM(AutoCausalLM):
    AUTO_MODEL_CLASS = AutoModelForCausalLM
    AutoCausalLM_ARGS = inspect.getfullargspec(AutoCausalLM.__init__).args
    def __init__(self, *args, **kwargs):
        self.bigdl_llm_kwargs = {}
        keys = list(kwargs.keys())
        for k in keys:
            if k not in self.AutoCausalLM_ARGS:
                self.bigdl_llm_kwargs[k] = kwargs[k]
                kwargs.pop(k)   
        AutoModelForCausalLM.from_pretrained = partial(AutoModelForCausalLM.from_pretrained, **self.bigdl_llm_kwargs)
        super().__init__(*args, **kwargs)

    @property
    def add_special_tokens(self) -> bool:
        return False

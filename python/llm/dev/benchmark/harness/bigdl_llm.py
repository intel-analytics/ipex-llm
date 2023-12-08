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
import multiprocessing

from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM

import inspect
from lm_eval.models.huggingface import AutoCausalLM
from functools import partial

class BigDLLM(AutoCausalLM):
    AutoCausalLM_ARGS = inspect.getfullargspec(AutoCausalLM.__init__).args
    def __init__(self, *args, **kwargs):
        self.bigdl_llm_kwargs = {}
        for k, v in kwargs.items():
            if k not in self.AutoCausalLM_ARGS:
                kwargs.pop(k)           
                self.bigdl_llm_kwargs[k] = v
        self.AUTO_MODEL_CLASS = partial(AutoModelForCausalLM, **self.bigdl_llm_kwargs)
        super().__init__(*args, **kwargs)
    
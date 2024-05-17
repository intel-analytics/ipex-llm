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
import importlib
import sys
from packaging import version

from ipex_llm.utils.common import invalidInputError
from enum import Enum

bigdl_patched = None  # None or 'Train' or 'Inference'
attrs = []


def replace_attr(obj, name: str, value):
    original_attr = getattr(obj, name)
    setattr(obj, name, value)
    attrs.append((obj, name, original_attr))


def llm_patch(train=False):
    '''
    llm_patch is used to make users' LLM application benefit from BigDL-LLM optimization
    with only one-line code patch.

    :param train: Whether to apply bigdl-llm patch for training code, default to be `False`.
    '''
    global bigdl_patched
    if bigdl_patched:
        return

    # Initial version of patch for llm finetuning, inference support TBD
    if train:
        from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
        replace_attr(transformers, "AutoModelForCausalLM", AutoModelForCausalLM)
        replace_attr(transformers, "LlamaForCausalLM", AutoModelForCausalLM)
        replace_attr(transformers, "AutoModel", AutoModel)
        from ipex_llm.transformers.utils import is_torch_bf16_gpu_available
        replace_attr(transformers.utils, "is_torch_bf16_gpu_available", is_torch_bf16_gpu_available)

        import_peft_check = 'peft' in sys.modules or 'peft.utils' in sys.modules or \
            'peft.tuners' in sys.modules or 'peft.mapping' in sys.modules
        invalidInputError(not import_peft_check,
                          'llm_patch() should be called at the beginning of your code.')
        import peft
        from ipex_llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training,\
            LoraConfig, TrainingArguments
        peft_version = peft.__version__
        replace_attr(transformers, "TrainingArguments", TrainingArguments)
        get_peft_model_original = getattr(peft, "get_peft_model")
        replace_attr(peft, "get_peft_model", get_peft_model)
        setattr(peft, "get_peft_model_original", get_peft_model_original)
        replace_attr(peft, "prepare_model_for_kbit_training", prepare_model_for_kbit_training)
        if version.parse(peft_version) <= version.parse("0.5.0"):
            replace_attr(peft, "prepare_model_for_int8_training", prepare_model_for_kbit_training)
        replace_attr(peft, "LoraConfig", LoraConfig)
        bigdl_patched = 'Train'


def llm_unpatch():
    '''
    llm_unpatch is an reverse function to llm_patch.
    '''
    global bigdl_patched

    if bigdl_patched is None:
        return

    for obj, name, torch_attr in attrs:
        setattr(obj, name, torch_attr)
    bigdl_patched = None

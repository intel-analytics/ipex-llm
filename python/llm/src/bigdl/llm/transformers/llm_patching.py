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
from bigdl.llm.utils.common import invalidInputError
from dataclasses import dataclass

is_bigdl_patched = False
patched_training_mode = None
attrs = []


def replace_attr(obj, name: str, value):
    original_attr = getattr(obj, name)
    setattr(obj, name, value)
    attrs.append((obj, name, original_attr))


def llm_patch(training_mode="qlora"):
    '''
    llm_patch is used to make users' LLM application benefit from BigDL-LLM optimization
    with only one-line code patch.

    :param training_mode: str, specify the training mode to be one of ["lora", "qlora", "qalora",
                          "relora"]. Default to be "qlora".
    '''
    global is_bigdl_patched
    global patched_training_mode
    patched_training_mode = training_mode
    if is_bigdl_patched:
        return

    # Initial version for LLM finetuning, other support TBD
    from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
    replace_attr(transformers, "AutoModelForCausalLM", AutoModelForCausalLM)
    replace_attr(transformers, "LlamaForCausalLM", AutoModelForCausalLM)
    replace_attr(transformers, "AutoModel", AutoModel)

    if importlib.util.find_spec("peft") is not None:
        import_peft_check = 'peft' in sys.modules or 'peft.utils' in sys.modules or \
            'peft.tuners' in sys.modules or 'peft.mapping' in sys.modules
        invalidInputError(not import_peft_check,
                          'llm_patch() should be called at the beginning of your code.')
        import peft
        from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training,\
            LoraConfig, TrainingArguments
        replace_attr(transformers, "TrainingArguments", TrainingArguments)
        get_peft_model_original = getattr(peft, "get_peft_model")
        replace_attr(peft, "get_peft_model", get_peft_model)
        setattr(peft, "get_peft_model_original", get_peft_model_original)
        replace_attr(peft, "prepare_model_for_kbit_training", prepare_model_for_kbit_training)
        replace_attr(peft, "prepare_model_for_int8_training", prepare_model_for_kbit_training)
        replace_attr(peft, "LoraConfig", LoraConfig)
        if training_mode == "relora":
            from bigdl.llm.transformers.relora import ReLoRATrainer
            replace_attr(transformers, "Trainer", ReLoRATrainer)

    is_bigdl_patched = True


def llm_unpatch():
    '''
    llm_unpatch is an reverse function to llm_patch.
    '''
    global is_bigdl_patched

    if not is_bigdl_patched:
        return

    for obj, name, torch_attr in attrs:
        setattr(obj, name, torch_attr)
    is_bigdl_patched = False

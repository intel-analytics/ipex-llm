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

is_bigdl_patched = False
attrs = []


def replace_attr(obj, name: str, value):
    original_attr = getattr(obj, name)
    setattr(obj, name, value)
    attrs.append((obj, name, original_attr))


def patch_bigdl_llm():
    '''
    patch_bigdl_llm is used to make users' LLM application benefit from BigDL-LLM optimization
    with only one-line code patch.
    '''
    global is_bigdl_patched
    if is_bigdl_patched:
        return

    # Initial version for LLM finetuning, other support TBD
    from bigdl.llm.transformers import AutoModelForCausalLM
    replace_attr(transformers, "AutoModelForCausalLM", AutoModelForCausalLM)

    if importlib.util.find_spec("peft") is not None:
        import peft
        from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training,\
            LoraConfig
        get_peft_model_original = getattr(peft, "get_peft_model")
        replace_attr(peft, "get_peft_model", get_peft_model)
        setattr(peft, "get_peft_model_original", get_peft_model_original)
        replace_attr(peft, "prepare_model_for_kbit_training", prepare_model_for_kbit_training)
        replace_attr(peft, "LoraConfig", LoraConfig)

    is_bigdl_patched = True


def unpatch_bigdl_llm():
    '''
    unpatch_bigdl_llm is an reverse function to patch_bigdl_llm.
    '''
    global is_bigdl_patched

    if not is_bigdl_patched:
        return

    for obj, name, torch_attr in attrs:
        setattr(obj, name, torch_attr)
    is_bigdl_patched = False

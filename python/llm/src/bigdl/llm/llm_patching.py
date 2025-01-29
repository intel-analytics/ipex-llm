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
import importlib
import sys
from bigdl.llm.utils.common import invalidInputError
from enum import Enum

bigdl_patched = None  # None or 'Train' or 'Inference'
attrs = []


def _parse_pretrained(am_fn, map={'device_map': None}):
    def mocked_am(self, *args, **kwargs):
        kwargs['device_map'] = map.pop('device_map', None)
        kwargs.update(map)
        return am_fn(self, *args, **kwargs)
    return mocked_am


def _parse_to(to_fn, map={'device': 'xpu'}):
    def mocked_to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device and 'cuda' in device.type:
            if kwargs.get('device', None):
                kwargs['device'] = map['device']
            else:
                args = list(args)
                args[0] = map['device']
        return to_fn(self, *args, **kwargs)
    return mocked_to


def replace_attr(obj, name: str, value):
    original_attr = getattr(obj, name)
    setattr(obj, name, value)
    attrs.append((obj, name, original_attr))


def llm_patch(train=False, device=None, load_in_low_bit=None):
    '''
    llm_patch is used to make users' LLM application benefit from BigDL-LLM optimization
    with only one-line code patch.

    :param train: Whether to apply bigdl-llm patch for training code, default to be `False`.
    '''
    global bigdl_patched
    if bigdl_patched:
        return

    from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel

    # patch bigdl pretrained
    am_map = dict(device_map=None, load_in_low_bit=load_in_low_bit)
    replace_attr(AutoModelForCausalLM, "from_pretrained",
                 _parse_pretrained(AutoModelForCausalLM.from_pretrained, am_map))
    replace_attr(AutoModel, "from_pretrained",
                 _parse_pretrained(AutoModel.from_pretrained, am_map))

    # patch transformers with bigdl
    replace_attr(transformers, "AutoModelForCausalLM", AutoModelForCausalLM)
    replace_attr(transformers, "LlamaForCausalLM", AutoModelForCausalLM)
    replace_attr(transformers, "AutoModel", AutoModel)

    # patch cuda with xpu
    if hasattr(torch, "xpu"):
        replace_attr(torch, "cuda", getattr(torch, "xpu"))
        replace_attr(torch.nn.Module, "cuda", getattr(torch.nn.Module, "xpu"))
        if not device:
            device = "xpu"
    replace_attr(torch.nn.Module, "to", _parse_to(torch.nn.Module.to, map={'device': 'xpu'}))
    if train:
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

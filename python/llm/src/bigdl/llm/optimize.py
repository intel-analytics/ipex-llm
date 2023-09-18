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

import torch
import os
import json
from .transformers import ggml_convert_low_bit
from torch.nn.modules import Module
from torch.nn.modules.module import _IncompatibleKeys
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError


# Simulate the Hugging Face format
PYTORCH_MODEL_NAME = "pytorch_model.bin"
CONFIG_NAME = "bigdl_config.json"


def _save_low_bit(self, save_dir, *args, **kwargs):
    invalidInputError(self._bigdl_config.get("bigdl_transformers_low_bit", False),
                      f"Detected this model is not a low-bit model, please use from_pretrained's"
                      f" load_in_4bit or load_in_low_bit parameter to load a 4-bit model first.")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, PYTORCH_MODEL_NAME)
    torch.save(self.state_dict(), model_path, *args, **kwargs)
    with open(os.path.join(save_dir, CONFIG_NAME), "w") as json_file:
        json.dump(self._bigdl_config, json_file)


# Under `init_empty_weights()`, we need to disable all actions
# that may lead to any parameter allocation", otherwise may need to error:
# NotImplementedError: Cannot copy out of meta tensor; no data!
class DisableTorchAllocTensor():
    def __init__(self) -> None:
        self._old_torch_load_state_dict = Module.load_state_dict
        self._old_torch_to_device = Module.to

    def __enter__(self):
        Module.load_state_dict = lambda *args, **kwargs: _IncompatibleKeys([], [])
        Module.to = lambda self, *args, **kwargs: self

    def __exit__(self, exc_type, exc_value, traceback):
        Module.load_state_dict = self._old_torch_load_state_dict
        Module.to = self._old_torch_to_device


def low_bit_sanity_check(model_path):
    invalidInputError(os.path.isdir(model_path),
                      "model_path should be a valid directory path.")
    invalidInputError(os.path.isfile(os.path.join(model_path, CONFIG_NAME)),
                      "bigdl_config.json should be under your model directory,"
                      "please check your input path.")
    with open(os.path.join(model_path, CONFIG_NAME), 'r') as f:
        _config = json.load(f)

    low_bit = _config.get("bigdl_transformers_low_bit", None)
    invalidInputError(low_bit,
                      "Detect this model is not a low-bit model, Please use `optimize_model`"
                      " with low_bit to get a low-bit model , and "
                      " serialize the model using save_low_bit first.")
    return low_bit


def load_low_bit(model_or_creator, model_path, **kwargs):
    is_creator = not isinstance(model_or_creator, torch.nn.Module) \
        and callable(model_or_creator)
    low_bit = low_bit_sanity_check(model_path)

    if low_bit:
        # a creator
        if is_creator:
            with init_empty_weights(), DisableTorchAllocTensor():
                model = model_or_creator(**kwargs)
        else:
            model = model_or_creator
        invalidInputError(isinstance(model, torch.nn.Module),
                          "model_or_creator should be a instance of "
                          "`torch.nn.Module`or a method that returns "
                          f"an instance of `torch.nn.Module`, but got {type(model)} at last.")
        qtype = ggml_tensor_qtype[low_bit]
        model = ggml_convert_low_bit(model, qtype=qtype, convert_shape_only=True)

    state_dict = torch.load(os.path.join(model_path, PYTORCH_MODEL_NAME))
    if is_creator:
        for param_name, param in state_dict.items():
            set_module_tensor_to_device(model, param_name, "cpu", param)
    else:
        model.load_state_dict(state_dict=state_dict)
    return model


def optimize_model(model, low_bit='sym_int4', optimize_llm=True):
    """
    A method to optimize any pytorch models.

    :param model: The original PyTorch model (nn.module)
    :param low_bit: Supported low-bit options are "sym_int4", "asym_int4", "sym_int5",
        "asym_int5" or "sym_int8".
    :param optimize_llm: Whether to further optimize llm model.

    return: The optimized model.
    """
    invalidInputError(low_bit in ggml_tensor_qtype,
                      f"Unknown load_in_low_bit value: {low_bit}, expected:"
                      f" sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8.")
    qtype = ggml_tensor_qtype[low_bit]
    model = ggml_convert_low_bit(model, qtype=qtype, optimize_model=optimize_llm)
    # add save_low_bit to pretrained model dynamically
    import types
    model._bigdl_config = dict()
    model._bigdl_config["bigdl_transformers_low_bit"] = low_bit
    model.save_low_bit = types.MethodType(_save_low_bit, model)
    return model

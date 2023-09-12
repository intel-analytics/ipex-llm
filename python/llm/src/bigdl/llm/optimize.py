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
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError


# Simulate the Hugging Face format
PYTORCH_MODEL_NAME = "pytorch_model.bin"
CONFIG_NAME = "bigdl_config.json"


def _save_low_bit(self, save_dir, *args, **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, PYTORCH_MODEL_NAME)
    torch.save(self.state_dict(), model_path, *args, **kwargs)
    with open(os.path.join(save_dir, CONFIG_NAME), "w") as json_file:
        json.dump(self._bigdl_config, json_file)


def load_low_bit(model, model_path):
    with open(os.path.join(model_path, CONFIG_NAME), 'r') as f:
        _config = json.load(f)
    
    low_bit = _config.get("bigdl_transformers_low_bit", None)

    if low_bit:
        qtype = ggml_tensor_qtype[low_bit]
        model = ggml_convert_low_bit(model, qtype=qtype, convert_shape_only=True)

    state_dict= torch.load(os.path.join(model_path, PYTORCH_MODEL_NAME))
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

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


# Some parts of this file is adapted from 
# https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/utils/bitsandbytes.py
# which is licensed under Apache License 2.0:
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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


import torch
import torch.nn as nn
from accelerate import init_empty_weights
from bigdl.llm.transformers.linear_4bit import Linear4bit, Params4bit
import warnings

def _replace_with_4bit_linear(model, modules_to_not_convert=None, current_key_name=None):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():

                    new_linear = Linear4bit(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                    )

                    # Copy the weights
                    new_linear._parameters['weight'] = Params4bit(data=module.weight.data,
                                                   requires_grad=False,
                                                   quantized=False,
                                                   _shape=None).to("cpu")
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data).to("cpu")

                    model._modules[name] = new_linear
                    has_been_replaced = True
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                    
        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_4bit_linear(
                module,
                modules_to_not_convert,
                current_key_name,
            )
    return model, has_been_replaced


def quantize_4bit(model):
    modules_to_not_convert = ["lm_head"]
    model, has_been_replaced = _replace_with_4bit_linear(
        model, modules_to_not_convert, None
    )
    if not has_been_replaced:
        warnings.warn(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " this can happen for some architectures such as gpt2 that uses Conv1D instead of Linear layers."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )
    return model

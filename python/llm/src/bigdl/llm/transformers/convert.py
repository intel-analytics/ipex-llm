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
# and https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
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
import warnings
import transformers
import importlib
from .utils import logger


def _replace_with_low_bit_linear(model, qtype, modules_to_not_convert=None,
                                 current_key_name=None, convert_shape_only=False):
    from bigdl.llm.transformers.low_bit_linear import LowBitLinear, FP4Params
    has_been_replaced = False

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    new_linear = LowBitLinear(
                        module.in_features,
                        module.out_features,
                        qtype,
                        module.bias is not None,
                    )

                    device_type = module.weight.data.device.type
                    # Copy the weights
                    paramsLowBit = FP4Params(data=module.weight.data,
                                             requires_grad=False,
                                             quantized=False,
                                             _shape=None,
                                             convert_shape_only=convert_shape_only,
                                             qtype=qtype).to(device_type)
                    new_linear._parameters['weight'] = paramsLowBit

                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device_type)

                    model._modules[name] = new_linear
                    has_been_replaced = True
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)

                    module.weight = None

        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            _, _flag = _replace_with_low_bit_linear(
                module,
                qtype,
                modules_to_not_convert,
                current_key_name,
                convert_shape_only,
            )
            has_been_replaced = _flag or has_been_replaced
    return model, has_been_replaced


def ggml_convert_low_bit(model, qtype, optimize_model=True,
                         convert_shape_only=False, device="cpu"):
    modules_to_not_convert = []  # ["lm_head"]
    model, has_been_replaced = _replace_with_low_bit_linear(
        model, qtype, modules_to_not_convert,
        None, convert_shape_only,
    )
    if not has_been_replaced:
        warnings.warn(
            "No linear modules were found in "
            "your model. This can happen for some architectures such as gpt2 that uses Conv1D "
            "instead of Linear layers. Please double check your model architecture, or submit "
            "an issue on github if you think this is a bug."
        )
    elif device == "cpu":
        model.to(torch.float32)
    elif device == "meta":
        # Do nothing here for weights are empty.
        pass

    if optimize_model:
        model = optimize(model)
    return model


def convert_forward(m, target_m, new_forward):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            bound_method = new_forward.__get__(sub_m, sub_m.__class__)
            setattr(sub_m, "forward", bound_method)
        convert_forward(sub_m, target_m, new_forward)


def optimize(model):
    from packaging import version
    from bigdl.llm.transformers.models.llama import llama_attention_forward_4_31
    from transformers.modeling_utils import PreTrainedModel

    # All huggingface format models are inherited from `PreTrainedModel`
    if not isinstance(model, PreTrainedModel):
        logger.info("Only HuggingFace Transformers models are currently "
                    "supported for further optimizations")
        return model

    trans_version = transformers.__version__
    if version.parse(trans_version) >= version.parse("4.31.0"):
        convert_forward(
            model,
            transformers.models.llama.modeling_llama.LlamaAttention,
            llama_attention_forward_4_31,)
    else:
        # todo implement 4.28.0 ~ 4.30.2
        pass

    if "chatglm2" in model.config._name_or_path:
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.chatglm2 import chatglm2_attention_forward_8eb45c
        from bigdl.llm.transformers.models.chatglm2 import core_attn_forward_8eb45c
        convert_forward(model,
                        module.SelfAttention,
                        chatglm2_attention_forward_8eb45c
                        )
        convert_forward(model,
                        module.CoreAttention,
                        core_attn_forward_8eb45c)
    elif "chatglm" in model.config._name_or_path:
        modeling_module_name = model.__class__.__module__
        module = importlib.import_module(modeling_module_name)
        from bigdl.llm.transformers.models.chatglm import chatglm_attention_forward
        convert_forward(model,
                        module.SelfAttention,
                        chatglm_attention_forward
                        )

    return model

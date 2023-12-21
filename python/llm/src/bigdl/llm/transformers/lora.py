
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
#
# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
# https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py
#
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import os
import sys
os.environ["ACCELERATE_USE_IPEX"] = "false"
os.environ["ACCELERATE_USE_XPU"] = "true"
from peft.tuners.lora import LoraLayer
from torch.nn import Linear


def patch_prepare_ipex(self, *args):
    return tuple(args)

# disable ipex.optimize
from accelerate import Accelerator
Accelerator._prepare_ipex = patch_prepare_ipex


# patch checkpoint function of torch
def _get_autocast_kwargs(device="xpu"):
    if device == "cuda":
        device_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    elif device == "xpu":
        device_autocast_kwargs = {
            "enabled": torch.xpu.is_autocast_xpu_enabled(),
            "dtype": torch.xpu.get_autocast_xpu_dtype(),
            "cache_enabled": False,
        }
    elif _supports_autocast(device):
        device_module = _get_device_module(device)
        device_autocast_kwargs = {
            "enabled": device_module.is_autocast_enabled(),
            "dtype": device_module.get_autocast_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    else:
        device_autocast_kwargs = None

    cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_cpu_enabled(),
        "dtype": torch.get_autocast_cpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    return device_autocast_kwargs, cpu_autocast_kwargs

def _supports_autocast(device):
    device_module = torch.utils.checkpoint._get_device_module(device)
    return device in ["xpu", "cuda"] or (hasattr(device_module, "is_autocast_enabled")
                                         and hasattr(device_module, "get_autocast_dtype"))

torch.utils.checkpoint._get_autocast_kwargs = _get_autocast_kwargs
torch.utils.checkpoint._supports_autocast = _supports_autocast


def cast_base_and_lora_weight(model, dtype=torch.bfloat16):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(dtype)
        if isinstance(module, Linear):
            module = module.to(dtype)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(dtype)


def preprare_gradient_checkpointing(model):
    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

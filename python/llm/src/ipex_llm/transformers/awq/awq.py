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
# ===========================================================================
#
# This file is adapted from
# https://github.com/casper-hansen/AutoAWQ/blob/main/awq/models/base.py#L147
#  and https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/quantizer.py
#
# MIT License
#
# Copyright (c) 2023 MIT HAN Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import gc
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers import AwqConfig, AutoConfig
from ipex_llm.transformers.awq.linear import WQLinear_GEMM, WQLinear_GEMV
from huggingface_hub import snapshot_download
from ipex_llm.utils.common import invalidInputError


layer_type_dict = {
    "mpt": "MPTBlock",
    "llama": "LlamaDecoderLayer",
    "opt": "OPTDecoderLayer",
    "RefinedWeb": "FalconDecoderLayer",
    "RefinedWebModel": "FalconDecoderLayer",
    "falcon": "FalconDecoderLayer",
    "bloom": "BloomBlock",
    "gptj": "GPTJBlock",
    "gpt_bigcode": "GPTBigCodeBlock",
    "mistral": "MistralDecoderLayer",
    "gpt_neox": "GPTNeoXDecoderLayer",
    "aquila": "AquilaDecoderLayer",
    "Yi": "YiDecoderLayer",
    "mixtral": "MixtralDecoderLayer",
}


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def _load_config(model_path, model_filename, safetensors=False,
                 trust_remote_code=True, max_new_tokens=4096):
    # [STEP 1] Download model if path is not a directory
    if not os.path.isdir(model_path):
        ignore_patterns = ["*msgpack*", "*h5*"]
        if safetensors:
            ignore_patterns.extend(["*.pt*", "*.bin*"])
        else:
            ignore_patterns.append("*.safetensors*")

        model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)

    if model_filename != '':
        model_weights_path = model_path + f'/{model_filename}'
    else:
        model_weights_path = model_path

    # Load model config and set max generation length
    max_new_tokens = 2048 if max_new_tokens is None else max_new_tokens
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config.max_new_tokens = max_new_tokens

    return model_weights_path, config


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == 'LlamaForCausalLM':
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif "mistral" in str(model.__class__).lower():
        layers = model.model.layers
    elif "yi" in str(model.__class__).lower():
        layers = model.model.layers
    elif "mixtral" in str(model.__class__).lower():
        layers = model.model.layers
    else:
        invalidInputError(False, f"Model type {type(model)} isn't supported.")
    return layers


def get_layer_type(config):
    if config.model_type not in layer_type_dict.keys():
        invalidInputError(False, f"{config.model_type} isn't supported yet.")
    return layer_type_dict[config.model_type]


def scale_activations(module):
    from ipex_llm.transformers.awq.act import ScaledActivation
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl,
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif 'mptblock' in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act,
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif 'falcon' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act,
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'bigcode' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act,
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'neox' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act,
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


def _replace_with_awq_layers(model, awq_config: AwqConfig):
    layers = get_blocks(model)

    for i in tqdm(range(len(layers)), desc="Replacing layers..."):
        layer = layers[i]

        # Get every linear layer in a block
        named_linears = get_named_linears(layer)

        # Replace activation functions
        scale_activations(layer)

        # Replace nn.Linear with WQLinear
        for name, module in named_linears.items():
            if any(key in name for key in awq_config.modules_to_not_convert):
                continue
            if awq_config.version == 'gemm':
                q_linear_module = WQLinear_GEMM
            elif awq_config.version == 'gemv':
                q_linear_module = WQLinear_GEMV

            q_linear = q_linear_module.from_linear(module,
                                                   awq_config.bits,
                                                   awq_config.group_size,
                                                   awq_config.backend,
                                                   True)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)

        gc.collect()

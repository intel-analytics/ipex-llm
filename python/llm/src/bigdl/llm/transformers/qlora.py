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
# https://github.com/huggingface/peft/blob/v0.5.0/src/peft/tuners/lora.py
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
# https://github.com/huggingface/peft/blob/v0.5.0/src/peft/tuners/lora.py
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
#
# Some parts of this file is adapted from
# https://github.com/catid/dora/blob/main/dora.py
#
# Copyright (c) 2024 Chris Taylor

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import logging
from torch.nn import Linear, Embedding
from bigdl.llm.transformers.low_bit_linear import LowBitLinear, BF16Linear, get_qk_size
from peft.tuners.lora import LoraLayer
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.utils import get_autocast_dtype
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
import functools
from bigdl.llm.transformers import training_patch
import torch.nn.functional as F

LOG = logging.getLogger("bigdl.llm.qlora")


class LoraLowBitLinear(LowBitLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        qa_lora: bool = True,
        **kwargs,
    ):
        LowBitLinear.__init__(
            self,
            in_features,
            out_features,
            qtype=kwargs.get("qtype"),
            bias=kwargs.get("bias", True),
            conver_to_half=False,
        )

        qk_size = get_qk_size(kwargs.get("qtype"))
        lora_in_features = in_features // qk_size if qa_lora else in_features
        LoraLayer.__init__(self, in_features=lora_in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        if qa_lora:
            self.qa_pool = torch.nn.AvgPool1d(qk_size)
        else:
            self.qa_pool = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        autocast_dtype = get_autocast_dtype(x)
        if x.device.type == "xpu":
            # force to use bf16 on gpu
            x = x.to(torch.bfloat16)
        elif autocast_dtype is not None:
            x = x.to(autocast_dtype)
        result = super().forward(x)

        if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter] > 0:
            result = result.clone()
            if autocast_dtype is None and x.device.type == "cpu":
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](
                            self.lora_dropout[self.active_adapter](self.qa_pool(x)))
                    ).to(expected_dtype)
                    * self.scaling[self.active_adapter]
                )
            else:
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](
                            self.lora_dropout[self.active_adapter](self.qa_pool(x)))
                    )
                    * self.scaling[self.active_adapter]
                )
            result += output
        return result


class LoraBF16Linear(BF16Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        BF16Linear.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            compute_dtype=torch.bfloat16,
        )

        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        autocast_dtype = get_autocast_dtype(x)
        if x.device.type == "xpu":
            # force to use bf16 on gpu
            x = x.to(torch.bfloat16)
        elif autocast_dtype is not None:
            x = x.to(autocast_dtype)
        result = super().forward(x)

        if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter] > 0:
            result = result.clone()
            if autocast_dtype is None and x.device.type == "cpu":
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](
                            self.lora_dropout[self.active_adapter](x))
                    ).to(expected_dtype)
                    * self.scaling[self.active_adapter]
                )
            else:
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](
                            self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )
            result += output
        return result


class DoraBF16Linear(BF16Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        BF16Linear.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            compute_dtype=torch.bfloat16,
        )

        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

        # Magnitude column-wise across output dimension
        self.m = self.weight.norm(p=2, dim=0, keepdim=True)
        self.fan_in_fan_out = kwargs.get("fan_in_fan_out", False)

    def forward(self, x: torch.Tensor):
        self.m = self.m.to(self.weight.device)
        from peft.utils import transpose
        delta_weight = (transpose(
            self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
            self.fan_in_fan_out,
        ) * self.scaling[self.active_adapter])
        adapted_weight = self.weight + delta_weight
        column_norm = adapted_weight.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted_weight / column_norm
        calc_weights = self.m * norm_adapted

        autocast_dtype = get_autocast_dtype(x)
        if x.device.type == "xpu":
            # force to use bf16 on gpu
            x = x.to(torch.bfloat16)
        elif autocast_dtype is not None:
            x = x.to(autocast_dtype)

        if calc_weights.dtype != x.dtype:
            calc_weights.data = calc_weights.data.to(x.dtype)
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # If x.shape>3, F.linear will use bmm, accounting for performance degradation.
        original_shape = x.shape
        # Convert to 2D shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])
        result = F.linear(x, self.weight, self.bias)
        # Convert to original shape
        if len(original_shape) > 2:
            result = result.reshape(*original_shape[:-1], result.shape[-1])

        return result.to(x.dtype)


def _create_new_module(create_new_module_func, lora_config, adapter_name, target, **kwargs):

    if isinstance(target, LowBitLinear) or isinstance(target, BF16Linear):
        low_bit_kwargs = kwargs.copy()
        bias = low_bit_kwargs.pop("bias", False)

        if hasattr(lora_config, "training_mode") and lora_config.training_mode == "lora":
            new_module = LoraBF16Linear(adapter_name,
                                        target.in_features,
                                        target.out_features,
                                        bias=bias,
                                        **low_bit_kwargs)
        elif hasattr(lora_config, "training_mode") and lora_config.training_mode == "dora":
            new_module = DoraBF16Linear(adapter_name,
                                        target.in_features,
                                        target.out_features,
                                        bias=bias,
                                        **low_bit_kwargs)
        else:
            if hasattr(lora_config, "training_mode"):
                qa_lora = lora_config.training_mode == "qalora"
            else:
                qa_lora = False
            low_bit_kwargs.update(
                {
                    "qtype": target.qtype,
                    "qa_lora": qa_lora
                }
            )
            new_module = LoraLowBitLinear(adapter_name,
                                          target.in_features,
                                          target.out_features,
                                          bias=bias,
                                          **low_bit_kwargs)
    else:
        new_module = create_new_module_func(lora_config, adapter_name, target, **kwargs)

    return new_module


from peft.tuners.lora import LoraModel
from peft.tuners.lora import LoraConfig as LoraConfigBase
from dataclasses import dataclass, field


@dataclass
class LoraConfig(LoraConfigBase):
    training_mode: str = field(default="qlora", metadata={"help": "determine training mode"})


def get_peft_model(*args, **kwargs):
    old_create_new_module = LoraModel._create_new_module
    LoraModel._create_new_module = staticmethod(functools.partial(_create_new_module,
                                                                  old_create_new_module))

    try:
        from peft import get_peft_model as get_peft_model_original
        model = get_peft_model_original(*args, **kwargs)
    finally:
        LoraModel._create_new_module = old_create_new_module

    if model.device.type == "xpu":
        cast_lora_weight(model, torch.bfloat16)
        _optimize_post(model)
        torch.xpu.synchronize()

    return model


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training.
    This includes:
        1- Cast the layernorm in fp32
        2- making output embedding layer require grads
        3- Add the upcasting of the lm head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """

    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if not is_gptq_quantized:
        # cast all non INT8 parameters to fp32
        # for param in model.parameters():
        #     if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
        #         param.data = param.data.to(torch.float32)

        # change to below way to reduce memory for Linear
        # otherwise lora finetuning on arc may OOM at this convert
        for module in model.modules():
            if list(module.children()) == []:
                # leaf module
                if not isinstance(module, (Linear, Embedding)):
                    for param in module.parameters():
                        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


class PeftModel:

    @staticmethod
    def from_pretrained(*args,
                        **kwargs):
        old_create_new_module = LoraModel._create_new_module
        LoraModel._create_new_module = staticmethod(functools.partial(_create_new_module,
                                                                      old_create_new_module))
        from peft import PeftModel
        try:
            model = PeftModel.from_pretrained(*args, **kwargs)
        finally:
            LoraModel._create_new_module = old_create_new_module

        return model


from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

PEFT_TYPE_TO_CONFIG_MAPPING["lora"] = LoraConfig


def cast_lora_weight(model, dtype=torch.bfloat16):
    for name, module in model.named_modules():
        if isinstance(module, LowBitLinear):
            module.compute_dtype = dtype
        if isinstance(module, LoraLayer):
            module = module.to(dtype)
        if isinstance(module, BF16Linear):
            module = module.to(dtype)
            module.compute_dtype = dtype
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(dtype)


def _optimize_post(model):
    import transformers
    from packaging import version
    from bigdl.llm.transformers.convert import convert_forward
    from bigdl.llm.transformers.models.llama import llama_attention_fast_forward

    trans_version = transformers.__version__
    if version.parse(trans_version) >= version.parse("4.31.0"):
        LOG.info("Optimizing Llama finetuning....")
        convert_forward(
            model,
            transformers.models.llama.modeling_llama.LlamaAttention,
            llama_attention_fast_forward,)

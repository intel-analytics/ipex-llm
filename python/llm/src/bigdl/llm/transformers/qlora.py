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
from bigdl.llm.transformers.low_bit_linear import LowBitLinear
from peft.tuners.lora import LoraLayer
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.utils import get_autocast_dtype
import functools


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
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    ).to(expected_dtype)
                    * self.scaling[self.active_adapter]
                )
            else:
                output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )
            result += output
        return result


def _create_new_module(create_new_module_func, lora_config, adapter_name, target, **kwargs):

    if isinstance(target, LowBitLinear):
        low_bit_kwargs = kwargs.copy()
        bias = low_bit_kwargs.pop("bias", False)
        low_bit_kwargs.update(
            {
                "qtype": target.qtype,
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


def get_peft_model(*args, **kwargs):
    old_create_new_module = LoraModel._create_new_module
    LoraModel._create_new_module = staticmethod(functools.partial(_create_new_module,
                                                                  old_create_new_module))
    try:
        from peft import get_peft_model as get_peft_model_original
        model = get_peft_model_original(*args, **kwargs)
    finally:
        LoraModel._create_new_module = old_create_new_module

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
        for param in model.parameters():
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


def patch_prepare_ipex(self, *args):
    return tuple(args)


from transformers.utils import (
    requires_backends,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_torch_xpu_available,
    is_sagemaker_dp_enabled,
    is_torch_tpu_available,
    is_torch_npu_available)
from transformers.utils.generic import strtobool
from transformers.utils import cached_property
from transformers.training_args import logger, ParallelMode, DistributedType
import torch
import torch.distributed as dist
import os
import warnings
from datetime import timedelta

if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState
    from accelerate.utils import DistributedType

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


@cached_property
def _setup_devices(self) -> "torch.device":
    requires_backends(self, ["torch"])
    logger.info("PyTorch: setting up devices")
    if not is_sagemaker_mp_enabled():
        if not is_accelerate_available(min_version="0.20.1"):
            invalidInputError(
                False,
                "Using the `Trainer` with `PyTorch` requires `accelerate>=0.20.1`: "
                "Please run `pip install transformers[torch]` or `pip install accelerate -U`"
            )
        AcceleratorState._reset_state(reset_partial_state=True)
    self.distributed_state = None
    if not self.use_ipex and "ACCELERATE_USE_IPEX" not in os.environ:
        os.environ["ACCELERATE_USE_IPEX"] = "false"
    if self.use_cpu or strtobool(os.environ.get("ACCELERATE_USE_CPU", "False")):
        self.distributed_state = PartialState(cpu=True, backend=self.ddp_backend)
        self._n_gpu = 0
    elif is_sagemaker_mp_enabled():
        local_rank = smp.local_rank()
        device = torch.device("cuda", local_rank)
        self._n_gpu = 1
        torch.cuda.set_device(device)
    elif is_torch_xpu_available() and "ACCELERATE_USE_XPU" not in os.environ:
        os.environ["ACCELERATE_USE_XPU"] = "true"
        self.distributed_state = PartialState(timeout=timedelta(seconds=self.ddp_timeout))
        # device = torch.device("xpu:0")
        device = self.distributed_state.device
        self._n_gpu = 1
    elif is_sagemaker_dp_enabled():
        self.distributed_state = PartialState(_use_sagemaker_dp=True)
        self._n_gpu = 1
    elif self.deepspeed:
        # Need to do similar for Accelerator init
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
        self.distributed_state = PartialState(timeout=timedelta(seconds=self.ddp_timeout))
        del os.environ["ACCELERATE_USE_DEEPSPEED"]
        self._n_gpu = 1
    else:
        self.distributed_state = PartialState(
            backend=self.ddp_backend, timeout=timedelta(seconds=self.ddp_timeout)
        )
        self._n_gpu = 1
    if not is_sagemaker_mp_enabled():
        device = self.distributed_state.device
        self.local_rank = self.distributed_state.local_process_index
    if dist.is_available() and dist.is_initialized() and \
            self.parallel_mode != ParallelMode.DISTRIBUTED:
        logger.warning(
            "torch.distributed process group is initialized, "
            "but parallel_mode != ParallelMode.DISTRIBUTED. "
            "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
        )
    if is_torch_tpu_available():
        device = self.distributed_state.device
        self._n_gpu = 0
    elif is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled():
        # Already set _n_gpu
        pass
    elif self.distributed_state.distributed_type == DistributedType.MULTI_XPU:
        if "ACCELERATE_USE_XPU" not in os.environ:
            os.environ["ACCELERATE_USE_XPU"] = "true"
        # self._n_gpu = torch.xpu.device_count()
        # device = torch.device("xpu:0")
        # torch.xpu.set_device(device)
    elif self.distributed_state.distributed_type == DistributedType.NO:
        if self.use_mps_device:
            warnings.warn(
                "`use_mps_device` is deprecated and will be removed in"
                " version 5.0 of 🤗 Transformers."
                "`mps` device will be used by default if available similar"
                " to the way `cuda` device is used."
                "Therefore, no action from user is required. "
            )
            if device.type != "mps":
                invalidInputError(False,
                                  ("Either you do not have an MPS-enabled device"
                                   " on this machine or MacOS"
                                   " version is not 12.3+ "
                                   "or current PyTorch install was not built with MPS enabled."))
        if device.type == "mps":
            self._n_gpu = 1
        elif self.use_cpu:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_xpu_available():
            device = torch.device("xpu:0")
            torch.xpu.set_device(device)
            self._n_gpu = 1
        elif is_torch_npu_available():
            device = torch.device("npu:0")
            torch.npu.set_device(device)
            self._n_gpu = 1
        else:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here,
            # so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
            if device.type == "cuda":
                torch.cuda.set_device(device)
    return device

# workaround a IPEX bug that prevents resume training in bf16
from accelerate import Accelerator
Accelerator._prepare_ipex = patch_prepare_ipex

# patch transformer for xpu DDP traing
from transformers import TrainingArguments
TrainingArguments._setup_devices = _setup_devices


def cast_lora_weight(model, dtype=torch.bfloat16):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(dtype)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(dtype)

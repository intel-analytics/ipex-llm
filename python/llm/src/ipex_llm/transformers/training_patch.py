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
#
# Some parts of this file is adapted from
# https://github.com/huggingface/peft/blob/v0.5.0/src/peft/config.py
# Copyright [yyyy] [name of copyright owner]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def patch_prepare_ipex(self, *args):
    return tuple(args)


from transformers.utils import (
    requires_backends,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_torch_xpu_available,
    is_peft_available,
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
from huggingface_hub import hf_hub_download
from ipex_llm.utils.common import invalidInputError

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

# remove ipex.optimize
from accelerate import Accelerator
Accelerator._prepare_ipex = patch_prepare_ipex

# patch transformer for xpu DDP traing
from transformers import TrainingArguments
TrainingArguments._setup_devices = _setup_devices

CONFIG_NAME = "adapter_config.json"


@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None, **kwargs):
    # Avoid circular dependency .. TODO: fix this with a larger refactor
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

    path = (
        os.path.join(pretrained_model_name_or_path, subfolder)
        if subfolder is not None
        else pretrained_model_name_or_path
    )

    hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

    if os.path.isfile(os.path.join(path, CONFIG_NAME)):
        config_file = os.path.join(path, CONFIG_NAME)
    else:
        try:
            config_file = hf_hub_download(pretrained_model_name_or_path, CONFIG_NAME,
                                          subfolder=subfolder, **hf_hub_download_kwargs)
        except Exception:
            invalidInputError(False,
                              f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

    loaded_attributes = cls.from_json_file(config_file)

    if "peft_type" in loaded_attributes:
        peft_type = loaded_attributes["peft_type"]
        config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
    else:
        config_cls = cls

    config = config_cls(**class_kwargs)

    for key, value in loaded_attributes.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config

# patch peft for merging adapter into the original model
if is_peft_available():
    from peft.config import PeftConfigMixin
    PeftConfigMixin.from_pretrained = from_pretrained

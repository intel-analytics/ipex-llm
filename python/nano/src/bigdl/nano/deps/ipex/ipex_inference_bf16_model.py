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

import contextlib
import subprocess
from logging import info, warning

from ...utils.log4Error import invalidInputError

from .ipex_inference_model import PytorchIPEXJITModel
from bigdl.nano.pytorch.amp.bfloat16 import autocast
import torch


class PytorchIPEXJITBF16Model(PytorchIPEXJITModel):
    def __init__(self, model, input_sample=None, use_ipex=False,
                 use_jit=False, channels_last=None, from_load=False):
        '''
        This is the accelerated model for pytorch and ipex/jit.
        All the external API is based on Trainer, so what we have here is
        basically internal APIs and subject to change.

        This PytorchIPEXJITModel will serve for fp32 and ipex>1.9 models.
        :param model: the model(nn.module) to be transform if from_load is False
               the accelerated model if from_load is True.
        :param input_sample: torch tensor indicate the data sample to be used
               for tracing.
        :param use_ipex: if use ipex to optimize the model
        :param use_jit: if use jit to accelerate the model
        :param channels_last: if set model and data to be channels-last mode.
               the parameter will be ignored if use_ipex is False.
        :param from_load: this will only be set by _load method.
        '''
        if use_ipex:
            invalidInputError(
                self._check_cpu_isa,
                errMsg="Applying IPEX BF16 optimization needs the cpu support avx512.",
                fixMsg="Please set use_ipex to False or not set precision to bf16."
            )
        PytorchIPEXJITModel.__init__(self, model, input_sample=input_sample, use_ipex=use_ipex,
                                     dtype=torch.bfloat16, use_jit=use_jit,
                                     channels_last=channels_last, from_load=from_load)

    @property
    def _check_cpu_isa(self):
        """Indicator to verify if cpu supports avx512"""
        msg = subprocess.check_output(["lscpu"]).decode("utf-8")
        return 'avx512' in msg or 'amx' in msg

    def autocast_context_manager(self):
        """Create autocast context"""
        return autocast(enabled=self._check_cpu_isa)

    @contextlib.contextmanager
    def forward_context(self):
        """Enable autocast context"""
        with self.autocast_context_manager():
            yield

    def forward_step(self, *inputs):
        with self.forward_context():
            return super().forward_step(*inputs)

    @property
    def status(self):
        status = super().status
        status.update({"precision": "bfloat16"})
        return status

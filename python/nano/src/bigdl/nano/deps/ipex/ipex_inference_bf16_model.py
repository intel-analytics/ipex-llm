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
import io
import os
from logging import info, warning

from ...utils.log4Error import invalidInputError

import intel_extension_for_pytorch as ipex
from .ipex_inference_model import PytorchIPEXJITModel
from bigdl.nano.pytorch.amp.bfloat16 import autocast, RedirectStream
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
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
        invalidInputError(
            self._has_bf16_isa or self._allow_non_bf16,
            errMsg="Your machine or OS doesn't support BF16 instructions."
        )
        dtype = torch.bfloat16 if self._has_bf16_isa else None
        PytorchIPEXJITModel.__init__(self, model, input_sample=input_sample, use_ipex=use_ipex,
                                     dtype=dtype, use_jit=use_jit,
                                     channels_last=channels_last, from_load=from_load)

    @property
    def _allow_non_bf16(self):
        """
        ALLOW_NON_BF16_ISA indicates if we restrict bf16 instructions support to be available.
        ALLOW_NON_BF16_ISA='1' sometimes helps debug and test cases without AVX512 or AMX

        :return: The bool value of ALLOW_NON_BF16_ISA
        """
        return os.environ.get("ALLOW_NON_BF16_ISA", None) == '1'

    @property
    def _has_bf16_isa(self):
        msg = subprocess.check_output(["lscpu"]).decode("utf-8")
        return 'avx512' in msg or 'amx' in msg

    def _max_bf16_isa(self, *args, **kwargs):
        # can not capture dnnl log
        dnnl_log = io.StringIO()
        with RedirectStream(target=dnnl_log), ipex.verbose(1):
            self.model(*args, *kwargs)
        dnnl_log = dnnl_log.getvalue()
        max_bf16_isa = None
        # IPEX 1.11 BF16 support AVX512
        # IPEX 1.12 BF16 support AMX, AVX512_BF16, AVX512_VNNI, AVX512
        if 'amx_bf16' in dnnl_log:
            max_bf16_isa = 'AMX'
        elif 'avx512_core_bf16' in dnnl_log:
            max_bf16_isa = 'AVX512_BF16'
        elif 'avx512_core' in dnnl_log:
            max_bf16_isa = 'AVX512'
        return max_bf16_isa

    def _bf16_check(self, *args, **kwargs):
        if getattr(self, "_is_bf16", None) is not None:
            return
        max_bf16_isa = self._max_bf16_isa(*args, **kwargs)
        if max_bf16_isa:
            info(f"{max_bf16_isa} BF16 support is enabled in this model.")
            self._is_bf16 = True
        else:
            if self._allow_non_bf16:
                self._is_bf16 = False
            else:
                invalidInputError(
                    False,
                    errMsg="BF16 ISA support is not enabled under current context.",
                    fixMsg="Please try to upgrade your pytorch version to obtain"
                           " BF16 acceleration."
                )

        if not self._is_bf16:
            warning("You are not running BF16 model with ISA support."
                    " The performance will be quite low.")

    def autocast_context_manager(self):
        return autocast(enabled=self._has_bf16_isa)

    @contextlib.contextmanager
    def forward_context(self):
        with self.autocast_context_manager():
            yield

    def forward_step(self, *inputs):
        with self.forward_context():
            if self.channels_last:
                inputs = tuple(map(lambda x: x.to(memory_format=torch.channels_last), inputs))
            self._bf16_check(*inputs)
            return self.model(*inputs)

    @property
    def status(self):
        status = super().status
        status.update({"precision": "bfloat16"})
        return status

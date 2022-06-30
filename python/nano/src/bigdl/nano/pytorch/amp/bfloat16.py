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
import io
from logging import info, warning

from pytorch_lightning import LightningModule
import torch
import subprocess
import os
from bigdl.nano.utils.log4Error import invalidOperationError, invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, LIGHTNING_VERSION_LESS_1_6


class autocast(torch.cpu.amp.autocast):
    """
    Customized autocast to verify BF16 support.
    """

    def __init__(self):
        super().__init__()

    def __enter__(self):
        self.global_max_cpu_isa = os.environ.get("ONEDNN_MAX_CPU_ISA", "ALL")
        os.environ["ONEDNN_MAX_CPU_ISA"] = "ALL"
        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.environ["ONEDNN_MAX_CPU_ISA"] = self.global_max_cpu_isa
        return super().__exit__(exc_type, exc_val, exc_tb)


class BF16Model(LightningModule):
    def __init__(self, model):
        super().__init__()
        invalidInputError(
            not TORCH_VERSION_LESS_1_10 and not LIGHTNING_VERSION_LESS_1_6,
            errMsg="Require torch>=1.10 and pytorch-lightning>=1.6.0."
        )
        self.bf16_model = model.bfloat16()

    @property
    def has_bf16_isa(self):
        msg = subprocess.check_output(["lscpu"]).decode("utf-8")
        return "avx512_core_bf16" in msg or "amx_bf16" in msg

    @autocast()
    def forward(self, *args, **kwargs):
        _is_bf16 = getattr(self, "_is_bf16", None)
        # ALLOW_NON_BF16_ISA indicates if we restrict bf16 instructions support to be available.
        # ALLOW_NON_BF16_ISA='1' sometimes helps debug and test cases without AVX512 or AMX
        allow_non_bf16 = os.environ.get("ALLOW_NON_BF16_ISA", None)
        if _is_bf16 is None:
            if self.has_bf16_isa:
                dnnl_log = io.StringIO()
                with contextlib.redirect_stdout(dnnl_log):
                    os.environ['DNNL_VERBOSE'] = '1'
                    self.bf16_model(*args, **kwargs)
                    os.environ['DNNL_VERBOSE'] = '0'
                dnnl_log = dnnl_log.getvalue()
                max_bf16_isa = None
                if 'amx_bf16' in dnnl_log:
                    max_bf16_isa = "AMX"
                elif 'avx512_core_bf16' in dnnl_log:
                    max_bf16_isa = "AVX512"
                if max_bf16_isa:
                    info("{} BF16 support is enabled in this run.".format(max_bf16_isa))
                    self._is_bf16 = True
                    _is_bf16 = True
                else:
                    self._is_bf16 = False
                    _is_bf16 = False
                    if allow_non_bf16 == '1':
                        self._is_bf16 = False
                        _is_bf16 = False
                    else:
                        invalidOperationError(
                            False,
                            errMsg="BF16 ISA support is not enabled under current context.",
                            fixMsg="Please try to upgrade your pytorch version to obtain"
                                   " BF16 acceleration."
                        )
            else:
                if allow_non_bf16 == '1':
                    self._is_bf16 = False
                    _is_bf16 = False
                else:
                    invalidOperationError(
                        False,
                        errMsg="Your machine or OS doesn't support BF16 instructions.",
                        fixMsg="Please check your machine and OS to make sure"
                               " BF16 support is available."
                    )

            if not _is_bf16:
                warning("You are not running BF16 model with ISA support."
                        " The performance will be quite low.")
        return self.bf16_model(*args, **kwargs)

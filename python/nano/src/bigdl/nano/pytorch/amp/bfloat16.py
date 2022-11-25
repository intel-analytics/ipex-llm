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
import sys
import fcntl

from pytorch_lightning import LightningModule
import torch
import subprocess
import os
from bigdl.nano.utils.inference.model import AcceleratedModel
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.log4Error import invalidOperationError, invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_12
from bigdl.nano.utils import CPUInfo

invalidInputError(
    not TORCH_VERSION_LESS_1_10,
    errMsg="Require torch>=1.10 to convert type as bfloat16."
)


class autocast(torch.cpu.amp.autocast):  # noqa
    """
    Customized autocast to verify BF16 support.
    """

    def __enter__(self):
        self.global_max_cpu_isa = os.environ.get("ONEDNN_MAX_CPU_ISA", "ALL")
        os.environ["ONEDNN_MAX_CPU_ISA"] = "ALL"
        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.environ["ONEDNN_MAX_CPU_ISA"] = self.global_max_cpu_isa
        return super().__exit__(exc_type, exc_val, exc_tb)


class RedirectStream(object):
    """Context manager to capture output of shared library"""
    def __init__(self, stream=sys.stdout, target=None):
        self.origin_stream = stream
        self.origin_stream_fileno = stream.fileno()
        self.target = io.StringIO() if target is None else target
        # Create a pipe to capture the stream
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        # Save a copy of the original stream
        self.origin_stream_fileno_dup = os.dup(self.origin_stream_fileno)
        # Replace the original stream with the write pipe
        os.dup2(self.pipe_in, self.origin_stream_fileno)
        os.close(self.pipe_in)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.origin_stream.flush()
        # Make pipe_out non-blocking
        fcntl.fcntl(self.pipe_out, fcntl.F_SETFL, os.O_NONBLOCK)
        while True:
            try:
                buf = os.read(self.pipe_out, 1024)
                if not buf:
                    break
                self.target.write(buf.decode('utf-8'))
            except OSError as e:
                break
        os.close(self.pipe_out)
        os.dup2(self.origin_stream_fileno_dup, self.origin_stream_fileno)
        os.close(self.origin_stream_fileno_dup)


@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    """
    Only redirect cpp stdout to file or os.devnull
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        # Python writes to fd
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            # allow code to be run with the redirected stdout
            yield
        finally:
            # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
            _redirect_stdout(to=old_stdout)


class BF16Model(AcceleratedLightningModule):
    """Model of BFloat16 with auto mixed precision."""

    def __init__(self, model, channels_last=None):  # noqa
        model.eval()
        super().__init__(model)
        self._bf16_check()
        self.model = model  # use mixed precision instead of complete precision
        self.channels_last = channels_last
        if self.channels_last is True:
            self.model = self.model.to(memory_format=torch.channels_last)

    @property
    def _has_bf16_isa(self):
        """Indicator to verify if bf16 instructions are available."""
        cpuinfo = CPUInfo()
        return cpuinfo.has_bf16

    @property
    def _allow_non_bf16(self):
        """
        ALLOW_NON_BF16_ISA indicates if we restrict bf16 instructions support to be available.
        ALLOW_NON_BF16_ISA='1' sometimes helps debug and test cases without AVX512 or AMX

        :return: The bool value of ALLOW_NON_BF16_ISA
        """
        return os.environ.get("ALLOW_NON_BF16_ISA", None) == '1'

    def _max_bf16_isa(self, *args, **kwargs):
        """
        Run inference once and check the log to confirm if bf16 instructions are used.

        :return:True/False
        """
        dnnl_log_file = "dnnl_log.log"
        with stdout_redirected(dnnl_log_file):
            os.environ['DNNL_VERBOSE'] = '1'
            self.bf16_model(*args, **kwargs)
        dnnl_log = ""
        with open(dnnl_log_file, "r") as f:
            dnnl_log = f.read()
        if os.path.exists(dnnl_log_file):
            os.remove(dnnl_log_file)
        max_bf16_isa = None
        if 'amx_bf16' in dnnl_log:
            max_bf16_isa = "AMX"
        elif 'avx512_core_bf16' in dnnl_log:
            max_bf16_isa = "AVX512"
        return max_bf16_isa

    def on_forward_start(self, inputs):
        return inputs

    @autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True)
    def forward_step(self, *inputs):
        if self.channels_last is True:
            inputs = tuple(map(lambda x: x.to(memory_format=torch.channels_last), inputs))
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        return outputs

    def _bf16_check(self):
        if getattr(self, "_is_bf16", None) is not None:
            return self._is_bf16

        invalidInputError(
            not TORCH_VERSION_LESS_1_12,
            errMsg="Require torch>=1.12 to obtain bfloat16 acceleration."
        )

        # ALLOW_NON_BF16_ISA indicates if we restrict bf16 instructions support to be available.
        # ALLOW_NON_BF16_ISA='1' sometimes helps debug and test cases without AVX512 or AMX
        if self._has_bf16_isa:
            self._is_bf16 = True
            # TODO: enable if torch >= 1.13,
            #  reference: https://github.com/pytorch/pytorch/commit/
            #  0e957465802204fb30e2a94cd330c16ba71955a6
            #  #diff-d730aecf3ceee9216948ee50d46f015c327d65b9f0c4981ef7adfa44dddc2673
            # max_bf16_isa = self._max_bf16_isa(*args, **kwargs)
            # if max_bf16_isa:
            #     info("{} BF16 support is enabled in this model.".format(max_bf16_isa))
            #     self._is_bf16 = True
            # else:
            #     if self._allow_non_bf16:
            #         self._is_bf16 = False
            #     else:
            #         invalidOperationError(
            #             False,
            #             errMsg="BF16 ISA support is not enabled under current context.",
            #             fixMsg="Please try to upgrade your pytorch version to obtain"
            #                    " BF16 acceleration."
            #         )
        else:
            # close error for no BF16 instructions, just warning.
            self._is_bf16 = False
            # if self._allow_non_bf16:
            #     self._is_bf16 = False

            # else:
            #     invalidOperationError(
            #         False,
            #         errMsg="Your machine or OS doesn't support BF16 instructions.",
            #         fixMsg="Please check your machine and OS to make sure"
            #                " BF16 support is available."
            #     )

        if not self._is_bf16:
            warning("Your machine or OS doesn't support BF16 instructions. "
                    "You are running BF16 model without ISA support, and the "
                    "performance might be quite low.")

    @property
    def status(self):
        status = super().status
        status.update({"channels_last": self.channels_last,
                       "checkpoint": "ckpt.pth"})
        return status

    @staticmethod
    def _load(path, model):
        status = BF16Model._load_status(path)
        checkpoint_path = path / status['checkpoint']
        state_dict = torch.load(checkpoint_path)
        model.eval()
        model.load_state_dict(state_dict)
        return BF16Model(model, channels_last=status['channels_last'])

    def _save_model(self, path):
        torch.save(self.model.state_dict(), path / "ckpt.pth")

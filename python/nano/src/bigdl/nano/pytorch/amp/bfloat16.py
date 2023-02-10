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


from logging import warning
import torch
import os
from bigdl.nano.utils.pytorch import generate_channels_last_available
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_12
from bigdl.nano.utils.common import _bf16_checker
from bigdl.nano.pytorch.context_manager import generate_context_manager

invalidInputError(
    not TORCH_VERSION_LESS_1_10,
    errMsg="Require torch>=1.10 to convert type as bfloat16."
)


class BF16Model(AcceleratedLightningModule):
    """Model of BFloat16 with auto mixed precision."""

    def __init__(self, model, input_sample=None, channels_last=None, channels_last_available=[], thread_num=None):  # noqa
        """
        This is the accelerated model for BFloat16 with auto mixed precision.

        :param model: the model(nn.module) to be transform.
        :param channels_last: if set model and data to be channels-last mode.
        :param channels_last_available: only passed by _load method,
               to decide which input can be converted to channels-last mode.
        :param thread_num: the thread num allocated for this model.
        """
        super().__init__(model)
        self._bf16_check()
        self.model = model  # use mixed precision instead of complete precision
        self.channels_last = channels_last
        self.thread_num = thread_num
        if self.channels_last is True:
            self.model = self.model.to(memory_format=torch.channels_last)
            if channels_last_available:  # init from load
                self.channels_last_available = channels_last_available
            else:  # init without channels_last_available loaded
                self.channels_last_available = generate_channels_last_available(input_sample)
        else:
            self.channels_last_available = []

        self._nano_context_manager = generate_context_manager(accelerator=None,
                                                              precision="bf16",
                                                              thread_num=thread_num)

    @property
    def _has_bf16_isa(self):
        """Indicator to verify if bf16 instructions are available."""
        return _bf16_checker()

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

    def __getattr__(self, name: str):
        # the search order is:
        # 1. current instance, like channels_last will be found at this place
        # 2. super class, like model will be found at this place
        # 3. original model, like additional attributes of original model
        #    will be found at this place
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        if self.channels_last is True:
            if self.channels_last_available:
                for idx, input in enumerate(inputs):
                    if self.channels_last_available[idx]:
                        input.to(memory_format=torch.channels_last)
            else:
                self.channels_last_available = generate_channels_last_available(inputs)
                for idx, input in enumerate(inputs):
                    if self.channels_last_available[idx]:
                        input.to(memory_format=torch.channels_last)
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
                       "channels_last_available": self.channels_last_available,
                       "checkpoint": "ckpt.pth",
                       "thread_num": self.thread_num})
        return status

    @staticmethod
    def _load(path, model):
        status = BF16Model._load_status(path)
        checkpoint_path = path / status['checkpoint']
        state_dict = torch.load(checkpoint_path)
        model.eval()
        model.load_state_dict(state_dict)
        thread_num = status.get('thread_num', None)
        if thread_num == {}:
            thread_num = None
        if thread_num is not None:
            thread_num = int(status['thread_num'])
        return BF16Model(model, channels_last=status['channels_last'],
                         channels_last_available=status['channels_last_available'],
                         thread_num=thread_num)

    def _save_model(self, path):
        torch.save(self.model.state_dict(), path / "ckpt.pth")

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

import torch
import operator
from bigdl.nano.utils.common import compare_version


class BaseContextManager(object):
    """
    No grad context manager for Pytorch Model Inference.

    This context manager is used for providing no_grad context only.
    """
    def __init__(self, thread_num=None, accelerator=None, enable_onednn=False):
        self.infer_mode_string = "inference_mode"
        self.infer_mode = torch.inference_mode(mode=True)
        self.thread_num = thread_num
        self.accelerator = accelerator
        self.enable_onednn = enable_onednn
        self.original_thread_num = torch.get_num_threads()

    def __enter__(self):
        if self.thread_num is not None:
            torch.set_num_threads(self.thread_num)
        if not hasattr(self, "infer_mode"):
            if self.infer_mode_string == "inference_mode":
                self.infer_mode = torch.inference_mode(mode=True)
            else:
                self.infer_mode = torch.no_grad()
        self.infer_mode.__enter__()
        if self.accelerator == "jit" and self.enable_onednn is True:
            if compare_version("torch", operator.ge, "1.12.0"):
                # onednn fusion be added to torch from version 1.12
                if not torch.jit.onednn_fusion_enabled():
                    torch.jit.enable_onednn_fusion(True)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.infer_mode.__exit__(exc_type, exc_value, exc_tb)
        if self.accelerator == "jit" and self.enable_onednn is True:
            if compare_version("torch", operator.ge, "1.12.0"):
                # onednn fusion be added to torch from version 1.12
                torch.jit.enable_onednn_fusion(False)
        torch.set_num_threads(self.original_thread_num)

    def __getstate__(self):
        state = self.__dict__
        if 'infer_mode' in state.keys():
            del state['infer_mode']
        return state


class AutocastContextManager(BaseContextManager):
    """
    Autocast context manager for Pytorch Model Inference.

    This context manager is used for providing no grad and autocast context,
    which is used for bf16 model.
    """
    def __init__(self, thread_num=None, accelerator=None, enable_onednn=False):
        super().__init__(thread_num=thread_num, accelerator=accelerator,
                         enable_onednn=enable_onednn)
        if compare_version("torch", operator.lt, "1.13.0"):
            # In torch1.12, torch.inference_mode(mode=True) will cause bug for jit+bf16
            self.infer_mode = torch.no_grad()
            self.infer_mode_string = "no_grad"
        self.autocast = torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)

    def __enter__(self):
        super().__enter__()
        if self.accelerator == "jit" and self.enable_onednn is True:
            if compare_version("torch", operator.lt, "1.14"):
                # onednn fusion for bf16 only work for torch version > 1.13
                if compare_version("torch", operator.ge, "1.12.0"):
                    # onednn fusion be added to torch from version 1.12
                    torch.jit.enable_onednn_fusion(False)
        if self.accelerator == "jit":
            # Disable AMP for JIT
            torch._C._jit_set_autocast_mode(False)
        self.autocast.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.autocast.__exit__(exc_type, exc_value, exc_tb)
        super().__exit__(exc_type, exc_value, exc_tb)


class XPUAutocastContextManager(AutocastContextManager):
    """
    Autocast context manager for Pytorch XPU Model Inference.

    This context manager is used for providing no grad and autocast context,
    which is used for bf16 model.
    """
    def __init__(self, thread_num=None, accelerator=None, enable_onednn=True):
        super().__init__(thread_num=thread_num, accelerator=accelerator,
                         enable_onednn=enable_onednn)
        self.autocast = torch.xpu.amp.autocast(enabled=True, dtype=torch.float16)

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        super().__exit__(exc_type, exc_value, exc_tb)


def generate_context_manager(accelerator=None, precision="fp32", thread_num=None,
                             enable_onednn=False, use_xpu=False):
    '''
    generate correct context manager according to different situation
    :param acclerator: str, the accelerator to use, we support "onnxruntime", "openvino",
           "jit", and None for pytorch framework.
    :param precision: str, the precision to use, we support "fp32", "bf16" and "int8".
    :param thread_num: int, the thread number to allocate, None for no limit.
    :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph
           API, which provides a flexible API for aggressive fusion. Default to
           ``False``, only valid when accelerator="jit", otherwise will be ignored.
    :param use_xpu: if xpu model is used.
    '''
    if (precision != "bf16" and use_xpu is False) or precision == "fp32":
        return BaseContextManager(thread_num=thread_num, accelerator=accelerator,
                                  enable_onednn=enable_onednn)
    elif not use_xpu:
        return AutocastContextManager(thread_num=thread_num, accelerator=accelerator,
                                      enable_onednn=enable_onednn)
    else:
        return XPUAutocastContextManager(thread_num=thread_num, accelerator=accelerator,
                                         enable_onednn=enable_onednn)

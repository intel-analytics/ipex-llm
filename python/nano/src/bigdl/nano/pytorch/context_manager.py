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


class BaseContextManager(object):
    """
    No grad context manager for Pytorch Model Inference.

    This context manager is used for providing no_grad context only.
    """

    def __init__(self, thread_num=None):
        self.no_grad = torch.no_grad()
        self.thread_num = thread_num
        self.original_thread_num = torch.get_num_threads()

    def __enter__(self):
        if self.thread_num is not None:
            torch.set_num_threads(self.thread_num)
        self.no_grad.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.no_grad.__exit__(exc_type, exc_value, exc_tb)
        torch.set_num_threads(self.original_thread_num)


class AutocastContextManager(BaseContextManager):
    """
    Autocast context manager for Pytorch Model Inference.

    This context manager is used for providing no grad and autocast context,
    which is used for bf16 model.
    """
    def __init__(self, thread_num=None):
        super().__init__(thread_num=thread_num)
        self.autocast = torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)

    def __enter__(self):
        super().__enter__()
        self.autocast.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.autocast.__exit__(exc_type, exc_value, exc_tb)
        super().__exit__(exc_type, exc_value, exc_tb)


def generate_context_manager(accelerator=None, precision="fp32", thread_num=None):
    '''
    generate correct context manager according to different situation
    :param acclerator: str, the accelerator to use, we support "onnxruntime", "openvino"
           and None for pytorch framework.
    :param precision: str, the precision to use, we support "fp32", "bf16" and "int8".
    :param thread_num: int, the thread number to allocate, None for no limit.
    '''
    if precision != "bf16":
        return BaseContextManager(thread_num=thread_num)
    else:
        return AutocastContextManager(thread_num=thread_num)

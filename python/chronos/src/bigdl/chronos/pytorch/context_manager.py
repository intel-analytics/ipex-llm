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
from bigdl.chronos.forecaster.utils import set_pytorch_thread


class DummyForecasterContextManager(object):
    """
    This context manager is used when users have enbaled `ForecasterContextManager`
    during inference
    """
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


class ForecasterContextManager(object):
    """
    This context manager is used for Pytorch Model Inference.

    Provide no grad, thread control and autocast context which is for bf16 model.
    """
    def __init__(self, forecaster, thread_num, optimize):
        self.forecaster = forecaster
        self.infer_mode = torch.inference_mode(mode=True)
        if thread_num:
            self.thread_num = thread_num
        elif optimize and self.forecaster.optimized_model_thread_num:
            self.thread_num = self.forecaster.optimized_model_thread_num
        else:
            self.thread_num = None
        self.bf16_enable = self.forecaster.accelerate_method is not None and \
            'bf16' in self.forecaster.accelerate_method
        if optimize and self.bf16_enable:
            self.autocast = torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)

    def __enter__(self):
        # call `torch.set_num_threads` at most once
        self.forecaster.thread_num = set_pytorch_thread(self.thread_num,
                                                        self.forecaster.thread_num)
        if hasattr(self.forecaster, 'optimized_model_thread_num'):
            self.forecaster.optimized_model_thread_num = \
                set_pytorch_thread(self.thread_num, self.forecaster.optimized_model_thread_num)
        self.forecaster.context_enabled = True
        self.infer_mode.__enter__()
        if self.bf16_enable:
            self.autocast.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.infer_mode.__exit__(exc_type, exc_value, exc_tb)
        if self.bf16_enable:
            self.autocast.__exit__(exc_type, exc_value, exc_tb)
        self.forecaster.context_enabled = False

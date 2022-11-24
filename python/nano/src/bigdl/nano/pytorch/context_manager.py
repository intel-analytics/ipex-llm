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

    no_grad = torch.no_grad()

    def __enter__(self):
        self.no_grad.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.no_grad.__exit__(exc_type, exc_value, exc_tb)


class AutocastContextManager(BaseContextManager):
    """
    Autocast context manager for Pytorch Model Inference.

    This context manager is used for providing no grad and autocast context,
    which is used for bf16 model.
    """

    autocast = torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)

    def __enter__(self):
        super().__enter__()
        self.autocast.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.autocast.__exit__(exc_type, exc_value, exc_tb)
        super().__exit__(exc_type, exc_value, exc_tb)

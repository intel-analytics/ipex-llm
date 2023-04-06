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


def apply_data_to_xpu(input_item):
    '''
    This function will apply xpu flag to
    the input item
    '''
    if torch.is_tensor(input_item):
        return input_item.to('xpu')
    return input_item


def apply_data_to_half(input_item):
    '''
    This function will apply xpu flag to
    the input item
    '''
    if torch.is_tensor(input_item):
        return input_item.half()
    return input_item

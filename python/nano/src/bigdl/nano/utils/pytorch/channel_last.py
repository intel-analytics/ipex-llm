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


def generate_channels_last_available(inputs):
    '''
    This function will generate a list of true and false to decide if the
    elements of input can be converted to channels_last
    '''
    # try channels_last available
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    if inputs is not None:  # to avoid the situation of inputs == None
        channels_last_available = [True] * len(inputs)
        for idx, input in enumerate(inputs):
            try:
                input.to(memory_format=torch.channels_last)
            except Exception as _e:
                channels_last_available[idx] = False
            else:
                channels_last_available[idx] = True
    else:
        channels_last_available = []
    return channels_last_available

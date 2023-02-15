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
    This function will generate a list of string to decide if the
    elements of input can be converted to

    channel_last: "channel_last"
    channel_last_3d: "channel_last_3d"
    no change: "original"
    '''
    # try channels_last available
    if inputs is not None:  # to avoid the situation of inputs == None
        if isinstance(inputs, torch.Tensor):
            inputs = tuple([inputs])  # make it an tuple for later process
        channels_last_available = ["original"] * len(inputs)
        for idx, input in enumerate(inputs):
            try:
                input.to(memory_format=torch.channels_last)
                channels_last_available[idx] = "channels_last"
            except Exception as _e:
                try:
                    input.to(memory_format=torch.channels_last_3d)
                    channels_last_available[idx] = "channels_last_3d"
                except Exception as _e:
                    pass
    else:
        channels_last_available = []
    return channels_last_available


def apply_proper_channels_last(flag, input_item):
    '''
    This function will apply proper channes_last to
    input item. flag has 3 possible values:

    channel_last: "channel_last"
    channel_last_3d: "channel_last_3d"
    no change: "original"
    '''
    if flag == "channels_last":
        return input_item.to(memory_format=torch.channels_last)
    if flag == "channels_last_3d":
        return input_item.to(memory_format=torch.channels_last_3d)
    return input_item

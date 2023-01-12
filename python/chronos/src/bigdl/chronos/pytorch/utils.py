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
import math
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader


def _inference(model, input_sample_list, batch_size):
    if batch_size is None:
        # this branch is only to speed up the inferencing when batch_size is set to None.
        return model(*input_sample_list).numpy()
    else:
        yhat_list = []
        sample_num = input_sample_list[0].shape[0]  # the first dim should be sample_num
        batch_num = math.ceil(sample_num / batch_size)
        for batch_id in range(batch_num):
            yhat_list.append(model(
                *tuple(map(lambda x: x[batch_id * batch_size: (batch_id + 1) * batch_size],
                           input_sample_list))).numpy())
        # this operation may cause performance degradation
        yhat = np.concatenate(yhat_list, axis=0)
        return yhat


def _pytorch_fashion_inference(model, input_data, batch_size=None):
    '''
    This is an internal inference pattern for any models which can be used like:
    `model(x)  # x is a pytorch tensor`

    :param model: The model to inference
    :param input_data: numpy ndarray
    :param batch_size: batch size

    :return: numpy ndarray
    '''
    with torch.no_grad():
        if isinstance(input_data, list):
            input_sample_list = list(map(lambda x: torch.from_numpy(x), input_data))
        elif isinstance(input_data, DataLoader):
            input_sample_list = [torch.cat(tuple(batch[0] for batch in input_data), dim=0)]
        else:
            input_sample_list = [torch.from_numpy(input_data)]
        yhat = _inference(model, input_sample_list, batch_size=batch_size)
    return yhat

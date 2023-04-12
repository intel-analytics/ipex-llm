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

import numpy as np
import torch
from ..common.log4Error import invalidInputError


def tensors_to_numpy(tensors):
    def to_numpy(ts):
        result = []
        for x in ts:
            if isinstance(x, torch.Tensor):
                result.append(x.cpu().detach().numpy())
            elif isinstance(x, np.ndarray):
                result.append(x)
            elif np.isscalar(x):
                # convert scalar to numpy too
                result.append(np.array(x))
            elif isinstance(x, Sequence):
                result.append(to_numpy(x))
            else:
                invalidInputError(False, f"Unexpected Type: {x}")
        return tuple(result)
    return to_numpy(tensors)


def cope_with_keyword_arguments(kwargs):
    # inplace convert kwargs
    for k in kwargs.keys():
        if isinstance(kwargs[k], tuple):
            kwargs[k] = numpy_to_tensors(kwargs[k])
        if isinstance(kwargs[k], torch.Tensor):
            kwargs[k] = kwargs[k].numpy()
        if np.isscalar(kwargs[k]):
            kwargs[k] = np.array(kwargs[k])

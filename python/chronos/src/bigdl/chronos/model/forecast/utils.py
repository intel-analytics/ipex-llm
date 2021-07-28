#
# Copyright 2018 Analytics Zoo Authors.
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
import random
import numpy
from torch.utils.data import TensorDataset, DataLoader


def np_to_creator(data):
    def data_creator(config, batch_size):
            return DataLoader(TensorDataset(torch.from_numpy(data[0]).float(),
                                            torch.from_numpy(data[1]).float()),
                              batch_size=batch_size,
                              shuffle=True)
    return data_creator


def set_pytorch_seed(seed):
    if seed is not None and isinstance(seed, int):
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

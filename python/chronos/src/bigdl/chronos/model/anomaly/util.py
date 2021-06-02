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

import numpy as np


def roll_arr(arr, stride):
    return np.asarray([arr[i:i + stride] for i in range(len(arr) - stride + 1)])


def scale_arr(arr, mode="minmax"):
    if mode == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaled = MinMaxScaler().fit_transform(arr).astype('float32')
    elif mode == "standard":
        from sklearn.preprocessing import StandardScaler
        scaled = StandardScaler().fit_transform(arr).astype('float32')
    else:
        raise ValueError("Unrecognized Mode")
    return scaled

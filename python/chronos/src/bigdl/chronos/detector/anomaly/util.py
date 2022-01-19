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
import logging


def sklearn_install_check():
    try:
        from sklearnex import patch_sklearn, unpatch_sklearn
        sklearnex_ailveable = True
    except ImportError:
        sklearnex_ailveable = False
        logging.warning("Please install sickit-learn-intelex first.")
    return sklearnex_ailveable


class CHR_DBSCAN:
    __slots__ = 'use_sklearnex', 'algorithm_list', 'sklearnex_ailveable', 'DBSCAN'

    def __init__(self, use_sklearnex, algorithm_list):
        self.use_sklearnex = use_sklearnex
        self.algorithm_list = algorithm_list
        self.sklearnex_ailveable = sklearn_install_check()

    def __enter__(self):
        if self.sklearnex_ailveable and self.use_sklearnex:
            from sklearnex import patch_sklearn
            patch_sklearn(self.algorithm_list)
        from sklearn.cluster import DBSCAN
        self.DBSCAN = DBSCAN
        return self.DBSCAN

    def __exit__(self, *args, **kwargs):
        if self.sklearnex_ailveable and self.use_sklearnex:
            from sklearnex import unpatch_sklearn
            unpatch_sklearn(self.algorithm_list)


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

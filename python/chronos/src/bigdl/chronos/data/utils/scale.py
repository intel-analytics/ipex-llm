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
import pandas as pd

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler


def _standard_scaler_unscale_timeseries_numpy(data, scaler, scaler_index):
    data_unscale = np.zeros(data.shape)
    feature_counter = 0
    for i in scaler_index:
        value_mean = scaler.mean_[i]
        value_scale = scaler.scale_[i]
        data_unscale[:, :, feature_counter] = data[:, :, feature_counter] * value_scale + value_mean
        feature_counter += 1
    return data_unscale

UNSCALE_HELPER_MAP = {StandardScaler: _standard_scaler_unscale_timeseries_numpy,
                      MaxAbsScaler: None,
                      MinMaxScaler: None,
                      RobustScaler: None}


def unscale_timeseries_numpy(data, scaler, scaler_index):
    return UNSCALE_HELPER_MAP[type(scaler)](data, scaler, scaler_index)

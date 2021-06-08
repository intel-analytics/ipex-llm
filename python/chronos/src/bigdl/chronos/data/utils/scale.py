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
        value_mean = scaler.mean_[i] if scaler.with_mean else 0
        value_scale = scaler.scale_[i] if scaler.with_std else 1
        data_unscale[:, :, feature_counter] = data[:, :, feature_counter] * value_scale + value_mean
        feature_counter += 1
    return data_unscale


def _maxabs_scaler_unscale_timeseries_numpy(data, scaler, scaler_index):
    data_unscale = np.zeros(data.shape)
    feature_counter = 0
    for i in scaler_index:
        value_max_abs = scaler.max_abs_[i]
        data_unscale[:, :, feature_counter] = data[:, :, feature_counter] * value_max_abs
        feature_counter += 1
    return data_unscale


def _minmax_scaler_unscale_timeseries_numpy(data, scaler, scaler_index):
    data_unscale = np.zeros(data.shape)
    feature_counter = 0
    for i in scaler_index:
        value_min = scaler.min_[i]
        value_scale = scaler.scale_[i]
        data_unscale[:, :, feature_counter] = \
            (data[:, :, feature_counter] - value_min) / value_scale
        feature_counter += 1
    return data_unscale


def _robust_scaler_unscale_timeseries_numpy(data, scaler, scaler_index):
    data_unscale = np.zeros(data.shape)
    feature_counter = 0
    for i in scaler_index:
        value_center = 0 if scaler.center_ is None else scaler.center_[i]
        value_scale = 1 if scaler.scale_ is None else scaler.scale_[i]
        data_unscale[:, :, feature_counter] = \
            data[:, :, feature_counter] * value_scale + value_center
        feature_counter += 1
    return data_unscale


UNSCALE_HELPER_MAP = {StandardScaler: _standard_scaler_unscale_timeseries_numpy,
                      MaxAbsScaler: _maxabs_scaler_unscale_timeseries_numpy,
                      MinMaxScaler: _minmax_scaler_unscale_timeseries_numpy,
                      RobustScaler: _robust_scaler_unscale_timeseries_numpy}


def unscale_timeseries_numpy(data, scaler, scaler_index):
    return UNSCALE_HELPER_MAP[type(scaler)](data, scaler, scaler_index)

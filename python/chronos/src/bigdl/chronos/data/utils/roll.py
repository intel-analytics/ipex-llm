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


def roll_timeseries_dataframe(df,
                              lookback,
                              horizon,
                              feature_col,
                              target_col):
    """
    roll dataframe into numpy ndarray sequence samples.
    :param input_df: a dataframe which has been resampled in uniform frequency.
    :param lookback: the length of the past sequence
    :param horizon: int or list,
           if `horizon` is an int, we will sample `horizon` step
           continuously after the forecasting point.
           if `horizon` is an list, we will sample discretely according
           to the input list. 1 means the timestampe just after the observed data.
    :param feature_col: list, indicate the feature col name.
    :param target_col: list, indicate the target col name.
    :return: x, y
        x: 3-d numpy array in format (no. of samples, lookback, feature_col length)
        y: 3-d numpy array in format (no. of samples, horizon, target_col length)
    Note: Specially, if `horizon` is set to 0, then there will not be y. (test mode)
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(lookback, int)
    assert isinstance(feature_col, list)
    assert isinstance(target_col, list)
    is_horizon_int = isinstance(horizon, int)
    is_horizon_list = isinstance(horizon, list) and\
        isinstance(horizon[0], int) and\
        min(horizon) > 0
    assert is_horizon_int or is_horizon_list

    is_test = True if (is_horizon_int and horizon == 0) else False
    if not is_test:
        return _roll_timeseries_dataframe_train(df,
                                                lookback,
                                                horizon,
                                                feature_col,
                                                target_col)
    else:
        return _roll_timeseries_dataframe_test(df,
                                               lookback,
                                               feature_col,
                                               target_col)


def _roll_timeseries_dataframe_test(df,
                                    lookback,
                                    feature_col,
                                    target_col):
    x = df.loc[:, feature_col+target_col].values

    output_x, mask_x = _roll_timeseries_ndarray(x, lookback)
    mask = (mask_x == 1)

    return output_x[mask]


def _roll_timeseries_dataframe_train(df,
                                     lookback,
                                     horizon,
                                     feature_col,
                                     target_col):
    max_horizon = horizon if isinstance(horizon, int) else max(horizon)
    x = df[:-max_horizon].loc[:, feature_col+target_col].values
    y = df.iloc[lookback:].loc[:, target_col].values

    output_x, mask_x = _roll_timeseries_ndarray(x, lookback)
    output_y, mask_y = _roll_timeseries_ndarray(y, horizon)
    mask = (mask_x == 1) & (mask_y == 1)

    return output_x[mask], output_y[mask]


def _roll_timeseries_ndarray(data, window):
    '''
    data should be a ndarray with num_dim = 2
    first dim is timestamp
    second dim is feature
    '''
    assert data.ndim == 2

    window_size = window if isinstance(window, int) else max(window)
    if isinstance(window, int):
        window_idx = np.arange(window)
    else:
        window_idx = np.array(window) - 1

    result = []
    mask = []
    for i in range(data.shape[0]-window_size+1):
        result.append(data[i+window_idx])
        if pd.isna(data[i+window_idx]).any(axis=None):
            mask.append(0)
        else:
            mask.append(1)

    return np.asarray(result), np.asarray(mask)

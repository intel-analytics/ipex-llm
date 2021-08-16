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
                              roll_feature_df,
                              lookback,
                              horizon,
                              feature_col,
                              target_col):
    """
    roll dataframe into numpy ndarray sequence samples.

    :param input_df: a dataframe which has been resampled in uniform frequency.
    :param roll_feature_df: an additional rolling feature dataframe that will
           be append to final result.
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
    Note: Specially, if `horizon` is set to 0, then y will be None.
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
                                                roll_feature_df,
                                                lookback,
                                                horizon,
                                                feature_col,
                                                target_col)
    else:
        return _roll_timeseries_dataframe_test(df,
                                               roll_feature_df,
                                               lookback,
                                               feature_col,
                                               target_col)


def _append_rolling_feature_df(rolling_result,
                               roll_feature_df):
    if roll_feature_df is None:
        return rolling_result
    additional_rolling_result = np.zeros((rolling_result.shape[0],
                                         rolling_result.shape[1],
                                         len(roll_feature_df.columns)))
    for idx in range(additional_rolling_result.shape[0]):
        for col_idx in range(additional_rolling_result.shape[2]):
            additional_rolling_result[idx, :, col_idx] = roll_feature_df.iloc[idx, col_idx]
    rolling_result = np.concatenate([rolling_result, additional_rolling_result], axis=2)
    return rolling_result


def _roll_timeseries_dataframe_test(df,
                                    roll_feature_df,
                                    lookback,
                                    feature_col,
                                    target_col):
    x = df.loc[:, target_col+feature_col].values.astype(np.float32)

    output_x, mask_x = _roll_timeseries_ndarray(x, lookback)
    mask = (mask_x == 1)

    x = _append_rolling_feature_df(output_x[mask], roll_feature_df)

    return x, None


def _roll_timeseries_dataframe_train(df,
                                     roll_feature_df,
                                     lookback,
                                     horizon,
                                     feature_col,
                                     target_col):
    max_horizon = horizon if isinstance(horizon, int) else max(horizon)
    x = df[:-max_horizon].loc[:, target_col+feature_col].values.astype(np.float32)
    y = df.iloc[lookback:].loc[:, target_col].values.astype(np.float32)

    output_x, mask_x = _roll_timeseries_ndarray(x, lookback)
    output_y, mask_y = _roll_timeseries_ndarray(y, horizon)
    mask = (mask_x == 1) & (mask_y == 1)

    x = _append_rolling_feature_df(output_x[mask], roll_feature_df)

    return x, output_y[mask]


def _shift(arr, num, fill_value=np.nan):
    # this function is adapted from
    # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def _roll_timeseries_ndarray(data, window):
    '''
    data should be a ndarray with num_dim = 2
    first dim is timestamp
    second dim is feature
    '''
    assert data.ndim == 2  # (num_timestep, num_feature)
    data = np.expand_dims(data, axis=1)  # (num_timestep, 1, num_feature)

    # window index and capacity
    window_size = window if isinstance(window, int) else max(window)
    if isinstance(window, int):
        window_idx = np.arange(window)
    else:
        window_idx = np.array(window) - 1

    roll_data = np.concatenate([_shift(data, i) for i in range(0, -window_size, -1)], axis=1)
    roll_data = roll_data[:data.shape[0]-window_size+1, window_idx, :]
    mask = ~np.any(np.isnan(roll_data), axis=(1, 2))

    return roll_data, mask

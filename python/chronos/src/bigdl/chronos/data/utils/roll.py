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
import pandas as pd


def roll_timeseries_dataframe(df,
                              roll_feature_df,
                              lookback,
                              horizon,
                              feature_col,
                              target_col,
                              id_col=None,
                              label_len=0,
                              contain_id=False,
                              deploy_mode=False):
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
           to the input list. 1 means the timestamp just after the observed data.
    :param feature_col: list, indicate the feature col name.
    :param target_col: list, indicate the target col name.
    :param id_col: str, indicate the id col name, only needed when contain_id is True.
    :param label_len: This parameter is only for transformer-based model.
    :param contain_id: This parameter is only for XShardsTSDataset
    :param deploy_mode: a bool indicates whether to use deploy mode, which will be used in
           production environment to reduce the latency of data processing. The value
           defaults to False.
    :return: x, y
        x: 3-d numpy array in format (no. of samples, lookback, feature_col length)
        y: 3-d numpy array in format (no. of samples, horizon, target_col length)
    Note: Specially, if `horizon` is set to 0, then y will be None.
    """
    if deploy_mode:
        return _roll_timeseries_dataframe_test(df,
                                               roll_feature_df,
                                               lookback,
                                               feature_col,
                                               target_col,
                                               id_col=id_col,
                                               contain_id=contain_id)

    from bigdl.nano.utils.log4Error import invalidInputError
    invalidInputError(isinstance(df, pd.DataFrame), "df is expected to be pandas dataframe")
    invalidInputError(isinstance(lookback, int), "lookback is expected to be int")
    invalidInputError(isinstance(feature_col, list), "feature_col is expected to be list")
    invalidInputError(isinstance(target_col, list), "target_col is expected to be list")
    is_horizon_int = isinstance(horizon, int)
    is_horizon_list = isinstance(horizon, list) and\
        isinstance(horizon[0], int) and\
        min(horizon) > 0
    invalidInputError(is_horizon_int or is_horizon_list,
                      "horizon is expected to be a list or int")

    # don't enter test mode if label_len!=0
    # TODO: change to use is_predict.
    is_test = True if (is_horizon_int and horizon == 0 and label_len == 0) else False
    if not is_test:
        return _roll_timeseries_dataframe_train(df,
                                                roll_feature_df,
                                                lookback,
                                                horizon,
                                                feature_col,
                                                target_col,
                                                id_col=id_col,
                                                label_len=label_len,
                                                contain_id=contain_id)
    else:
        return _roll_timeseries_dataframe_test(df,
                                               roll_feature_df,
                                               lookback,
                                               feature_col,
                                               target_col,
                                               id_col=id_col,
                                               contain_id=contain_id)


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
                                    target_col,
                                    id_col,
                                    contain_id):
    x = df.loc[:, target_col+feature_col].values.astype(np.float32)

    output_x, mask_x = _roll_timeseries_ndarray(x, lookback)
    mask = (mask_x == 1)

    x = _append_rolling_feature_df(output_x[mask], roll_feature_df)

    if contain_id:
        return x, None, df.loc[:, [id_col]].values
    else:
        return x, None


def _roll_timeseries_dataframe_train(df,
                                     roll_feature_df,
                                     lookback,
                                     horizon,
                                     feature_col,
                                     target_col,
                                     id_col,
                                     label_len,
                                     contain_id):
    from bigdl.nano.utils.log4Error import invalidInputError
    if label_len != 0 and isinstance(horizon, list):
        invalidInputError(False,
                          "horizon should be an integer if label_len is set to larger than 0.")
    max_horizon = horizon if isinstance(horizon, int) else max(horizon)
    if max_horizon > 0:
        x = df[:-max_horizon].loc[:, target_col+feature_col].values.astype(np.float32)
    else:
        x = df.loc[:, target_col+feature_col].values.astype(np.float32)
    y = df.iloc[lookback-label_len:].loc[:, target_col].values.astype(np.float32)

    output_x, mask_x = _roll_timeseries_ndarray(x, lookback)
    if isinstance(horizon, list):
        output_y, mask_y = _roll_timeseries_ndarray(y, horizon)
    else:
        output_y, mask_y = _roll_timeseries_ndarray(y, horizon+label_len)
    mask = (mask_x == 1) & (mask_y == 1)

    x = _append_rolling_feature_df(output_x[mask], roll_feature_df)

    if contain_id:
        return x, output_y[mask], df.loc[:, [id_col]].values
    else:
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
    from bigdl.nano.utils.log4Error import invalidInputError
    invalidInputError(data.ndim == 2,
                      "data dim is expected to be 2")  # (num_timestep, num_feature)
    data = np.expand_dims(data, axis=1)  # (num_timestep, 1, num_feature)

    # window index and capacity
    window_size = window if isinstance(window, int) else max(window)
    if isinstance(window, int):
        window_idx = np.arange(window)
    else:
        window_idx = np.array(window) - 1

    roll_data = np.concatenate([_shift(data, i) for i in range(0, -window_size, -1)], axis=1)
    if data.shape[0] >= window_size:
        roll_data = roll_data[:data.shape[0]-window_size+1, window_idx, :]
    else:
        roll_data = roll_data[:0, window_idx, :]  # no sample will be sampled
    mask = ~np.any(np.isnan(roll_data), axis=(1, 2))

    return roll_data, mask

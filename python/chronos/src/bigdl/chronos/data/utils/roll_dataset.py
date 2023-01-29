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
from torch.utils.data import Dataset
import torch
import pandas as pd

from bigdl.chronos.data.utils.utils import _check_cols_no_na, _to_list
from bigdl.chronos.data.utils.time_feature import time_features, gen_time_enc_arr


def get_roll_start_idx(df, id_col, window_size):
    import itertools
    if not id_col:
        id_start_idxes = [0, len(df.index)]
    else:
        id_start_idxes = df.index[df[id_col] != df[id_col].shift(1)].tolist() + [len(df.index)]
    roll_start_idx_iter = ((range(id_start_idxes[i], id_start_idxes[i+1] - window_size + 1))
                           for i in range(len(id_start_idxes) - 1))
    roll_start_idxes = np.fromiter(itertools.chain.from_iterable(roll_start_idx_iter), np.int)
    return roll_start_idxes


class RollDataset(Dataset):
    def __init__(self, df, dt_col, freq, lookback, horizon, feature_col, target_col,
                 id_col=None, time_enc=False, label_len=0, is_predict=False):
        """
        A customized TorchDataset for rolling dataframe for time series applications.

        :param df: The dataframe to roll on. The dataframe could contain single id value or
            multiple id values. If the dataframe contains multiple ids, the rows of same id
            should be consecutive. And dataframe should have been ordered by timestamp for each id.
        :param dt_col: a str indicates the col name of datetime column in the input data frame, the
            dt_col must be sorted from past to latest respectively for each id.
        :param freq: The freq(interval) of this dataset, it is normal if freq is None,
            which means that we can not determine the frequency of this dataset. This parameter is
            needed when time_enc is True(for autoformer).
        :param lookback: the length of the past sequence
        :param horizon: int or list,
           if `horizon` is an int, we will sample `horizon` step
           continuously after the forecasting point.
           if `horizon` is an list, we will sample discretely according
           to the input list. 1 means the timestamp just after the observed data.
        :param feature_col: list, indicate the feature col name.
        :param target_col: list, indicate the target col name.
        :param id_col: (optional) a str indicates the col name of dataframe id.
        :param time_enc: bool,
               This parameter should be set to True only when you are using Autoformer model. With
               time_enc to be true, 2 additional numpy ndarray will be returned when you call
               `.to_numpy()`. Be sure to have a time type for dt_col if you set time_enc to True.
        :param label_len: int,
               This parameter should be set to True only when you are using Autoformer model. This
               indicates the length of overlap area of output(y) and input(x) on time axis.
        :param is_predict: bool,
               This parameter indicates if the dataset will be sampled as a prediction dataset
               (without groud truth).

        :return:

        """
        # horizon_time is only for time_enc, the time_enc numpy ndarray won't have any
        # shape change when the dataset is for prediction.
        self.horizon_time = horizon
        if horizon == 0:
            is_predict = True
        if is_predict:
            horizon = 0

        df.reset_index(drop=True, inplace=True)
        feature_col = _to_list(feature_col, "feature_col")
        target_col = _to_list(target_col, "target_col")
        _check_cols_no_na(df, col_names=target_col + feature_col)
        cols = target_col + feature_col
        cols = cols[0] if len(cols) == 1 else cols
        self.arr = df.loc[:, cols].to_numpy()
        self.arr = np.expand_dims(self.arr, axis=1) if self.arr.ndim == 1 else self.arr
        max_horizon = horizon if isinstance(horizon, int) else max(horizon)
        window_size = lookback + max_horizon
        self.roll_start_idxes = get_roll_start_idx(df, id_col, window_size=window_size)
        self.lookback = lookback
        self.horizon = horizon
        self.target_num = len(target_col)
        self.is_predict = is_predict

        # time_enc
        self.time_enc = time_enc
        self.label_len = label_len
        if self.time_enc:
            df_stamp = pd.DataFrame(columns=[dt_col])
            if is_predict:
                pred_dates = pd.date_range(df[dt_col].values[-1], periods=self.horizon_time + 1,
                                           freq=freq)
                df_stamp.loc[:, dt_col] = list(df[dt_col].values) + list(pred_dates[1:])
            else:
                df_stamp.loc[:, dt_col] = list(df[dt_col].values)
            data_stamp = time_features(pd.to_datetime(df_stamp[dt_col].values), freq=freq)
            self.data_stamp_arr = data_stamp.transpose(1, 0)

    def __len__(self):
        return self.roll_start_idxes.size

    def __getitem__(self, idx):
        start_idx = self.roll_start_idxes[idx]

        # cal x
        x = self.arr[start_idx: start_idx + self.lookback]
        x = torch.from_numpy(x).float()
        if self.is_predict is True and not self.time_enc:
            return x

        # cal y
        arr_target_only = self.arr[:, :self.target_num]
        if isinstance(self.horizon, int):
            y = arr_target_only[start_idx + self.lookback - self.label_len:
                                start_idx + self.lookback + self.horizon]
        else:
            # horizon is a list of int
            horizons = np.array(self.horizon)
            y = np.take(arr_target_only, horizons + start_idx + self.lookback - 1, axis=0)
        y = torch.from_numpy(y).float()

        if self.time_enc:
            # cal x_enc
            x_enc = self.data_stamp_arr[start_idx: start_idx + self.lookback]
            x_enc = torch.from_numpy(x_enc).float()
            # cal y_enc
            y_enc = self.data_stamp_arr[start_idx + self.lookback - self.label_len:
                                        start_idx + self.lookback + self.horizon_time]
            y_enc = torch.from_numpy(y_enc).float()

        if self.time_enc:
            return x, y, x_enc, y_enc
        else:
            return x, y

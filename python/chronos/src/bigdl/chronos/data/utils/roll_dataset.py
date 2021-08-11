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
from torch.utils.data import Dataset
import torch

from zoo.chronos.data.utils.utils import _check_cols_no_na, _to_list


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
    def __init__(self, df, lookback, horizon, feature_col, target_col, id_col=None):
        """
        A customized TorchDataset for rolling dataframe for time series applications.

        :param df: The dataframe to roll on. The dataframe could contain single id value or
            multiple id values. If the dataframe contains multiple ids, the rows of same id
            should be consecutive. And dataframe should have been ordered by timestamp for each id.
        :param lookback: the length of the past sequence
        :param horizon: int or list,
           if `horizon` is an int, we will sample `horizon` step
           continuously after the forecasting point.
           if `horizon` is an list, we will sample discretely according
           to the input list. 1 means the timestamp just after the observed data.
        :param feature_col: list, indicate the feature col name.
        :param target_col: list, indicate the target col name.
        :param id_col: (optional) a str indicates the col name of dataframe id

        :return:

        """
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

    def __len__(self):
        return self.roll_start_idxes.size

    def __getitem__(self, idx):
        start_idx = self.roll_start_idxes[idx]
        x = self.arr[start_idx: start_idx + self.lookback]
        x = torch.from_numpy(x).float()
        if self.horizon == 0:
            return x

        # cal y
        arr_target_only = self.arr[:, :self.target_num]
        if isinstance(self.horizon, int):
            y = arr_target_only[start_idx + self.lookback: start_idx + self.lookback + self.horizon]
        else:
            # horizon is a list of int
            horizons = np.array(self.horizon)
            y = np.take(arr_target_only, horizons + start_idx + self.lookback - 1, axis=0)
        y = torch.from_numpy(y).float()
        return x, y

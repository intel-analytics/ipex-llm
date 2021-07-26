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

import pandas as pd


def resample_timeseries_dataframe(df,
                                  dt_col,
                                  interval,
                                  start_time=None,
                                  end_time=None,
                                  id_col=None,
                                  merge_mode="mean"):
    '''
    resample and return a dataframe with a new time interval.
    :param df: input dataframe.
    :param dt_col: name of datetime column.
    :param interval: pandas offset aliases, indicating time interval of the output dataframe
    :param start_time: start time of the output dataframe
    :param end_time: end time of the output dataframe
    :param id_col: name of id column, this column won't be resampled
    :param merge_mode: if current interval is smaller than output interval,
        we need to merge the values in a mode. "max", "min", "mean"
        or "sum" are supported for now.
    '''
    assert dt_col in df.columns, f"dt_col {dt_col} can not be found in df."
    assert pd.isna(df[dt_col]).sum() == 0, "There is N/A in datetime col"
    assert merge_mode in ["max", "min", "mean", "sum"],\
        f"merge_mode should be one of [\"max\", \"min\", \"mean\", \"sum\"]," \
        f" but found {merge_mode}."

    res_df = df.copy()
    id_name = None
    if id_col:
        id_name = res_df[id_col].iloc[0]
        res_df.drop(id_col, axis=1)
    res_df.set_index(dt_col, inplace=True)
    res_df = res_df.resample(pd.Timedelta(interval))

    if merge_mode == "max":
        res_df = res_df.max()
    elif merge_mode == "min":
        res_df = res_df.min()
    elif merge_mode == "mean":
        res_df = res_df.mean()
    elif merge_mode == "sum":
        res_df = res_df.sum()

    start_time_stamp = pd.Timestamp(start_time) if start_time else res_df.index[0]
    end_time_stamp = pd.Timestamp(end_time) if end_time else res_df.index[-1]
    assert start_time_stamp <= end_time_stamp, "end time must be later than start time."

    offset = (start_time_stamp - res_df.index[0]) % pd.Timedelta(interval)
    new_index = pd.date_range(start=start_time_stamp-offset, end=end_time_stamp, freq=interval)
    res_df = res_df.reindex(new_index)
    res_df.index.name = dt_col
    res_df = res_df.reset_index()
    if id_col:
        res_df[id_col] = id_name
    return res_df

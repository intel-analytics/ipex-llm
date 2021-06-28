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
                                  merge_mode="mean"):
    '''
    resample and return a dataframe with a new time interval.
    :param df: input dataframe.
    :param dt_col: name of datetime column.
    :param interval: pandas offset aliases, indicating time interval of the output dataframe
    :param start_time: start time of the output dataframe
    :param end_time: end time of the output dataframe
    :param merge_mode: if current interval is smaller than output interval,
        we need to merge the values in a mode. "max", "min", "mean"
        or "sum" are supported for now.
    '''
    assert dt_col in df.columns, f"dt_col {dt_col} can not be found in df."
    assert pd.isna(df[dt_col]).sum() == 0, "There is N/A in datetime col"
    assert merge_mode in ["max", "min", "mean", "sum"],\
        f"merge_mode should be one of [\"max\", \"min\", \"mean\", \"sum\"]," \
        f" but found {merge_mode}."

    start_time_stamp = pd.Timestamp(start_time) if start_time else df[dt_col].iloc[0]
    end_time_stamp = pd.Timestamp(end_time) if end_time else df[dt_col].iloc[-1]
    zero_time_stamp = pd.Timestamp(0, unit='ms')
    assert start_time_stamp <= end_time_stamp, "end time must be later than start time."
    res_df = df.copy()
    res_df[dt_col] = df.apply(
        lambda row: resample_helper(
            row[dt_col],
            interval,
            start_time_stamp,
            end_time_stamp,
            zero_time_stamp),
        axis=1)
    res_df = res_df[~res_df[dt_col].isin([None])]

    if merge_mode == "max":
        res_df = res_df.groupby([dt_col]).max()
    if merge_mode == "min":
        res_df = res_df.groupby([dt_col]).min()
    if merge_mode == "mean":
        res_df = res_df.groupby([dt_col]).mean()
    if merge_mode == "sum":
        res_df = res_df.groupby([dt_col]).sum()

    new_start = start_time_stamp + \
        (interval - divmod(start_time_stamp - zero_time_stamp, pd.Timedelta(interval))[1])
    new_end = end_time_stamp - \
        divmod(end_time_stamp - zero_time_stamp, pd.Timedelta(interval))[1]
    new_end = new_start if new_start > new_end else new_end
    new_index = pd.date_range(start=new_start, end=new_end, freq=interval)
    res_df = res_df.reindex(new_index)
    res_df.index.name = dt_col
    res_df = res_df.reset_index()
    return res_df


def resample_helper(curr_time,
                    interval,
                    start_time_stamp,
                    end_time_stamp,
                    zero_time_stamp):
    offset = divmod((curr_time - zero_time_stamp), pd.Timedelta(interval))[1]
    if(offset / interval) >= 0.5:
        resampled_time = curr_time + (interval - offset)
    else:
        resampled_time = curr_time - offset
    if (resampled_time < start_time_stamp or resampled_time > end_time_stamp):
        return None
    return resampled_time

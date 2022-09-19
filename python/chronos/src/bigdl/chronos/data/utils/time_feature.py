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
# MIT License

# Copyright (c) 2021 THUML @ Tsinghua University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# code adapted from https://github.com/thuml/Autoformer
from typing import List

import numpy as np
import pandas as pd
from pandas import Timedelta
from pandas.tseries.frequencies import to_offset
from bigdl.chronos.data.utils.roll import _roll_timeseries_ndarray


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(offset) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = (
        (Timedelta(seconds=60), [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear]),  # 6 for second - minutes
        (Timedelta(minutes=60), [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ]),  # 5 for minutes - hour
        (Timedelta(hours=24), [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]),  # 4 for hour - day
        (Timedelta(days=7), [DayOfWeek, DayOfMonth, DayOfYear]),  # 3 for day - week
        (Timedelta(days=30), [DayOfMonth, WeekOfYear]),  # 2 for week - month
        (Timedelta(days=365), [MonthOfYear]),  # 1 for month - year
    )

    for offset_type, feature_classes in features_by_offsets:
        if offset < offset_type:
            return [cls() for cls in feature_classes]
    return []  # freq larger than 1 year


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


def gen_time_enc_arr(df, dt_col, freq, horizon_time, is_predict, lookback, label_len):
    df_stamp = pd.DataFrame(columns=[dt_col])
    if is_predict:
        pred_dates = pd.date_range(df[dt_col].values[-1],
                                   periods=horizon_time + 1, freq=freq)
        df_stamp.loc[:, dt_col] =\
            list(df[dt_col].values) + list(pred_dates[1:])
    else:
        df_stamp.loc[:, dt_col] = list(df[dt_col].values)
    data_stamp = time_features(pd.to_datetime(df_stamp[dt_col].values),
                               freq=freq)
    data_stamp = data_stamp.transpose(1, 0)
    max_horizon = horizon_time if isinstance(horizon_time, int) else max(horizon_time)
    numpy_x_timeenc, _ = _roll_timeseries_ndarray(data_stamp[:-max_horizon],
                                                  lookback)
    numpy_y_timeenc, _ = _roll_timeseries_ndarray(data_stamp[lookback-label_len:],
                                                  horizon_time+label_len)
    return numpy_x_timeenc, numpy_y_timeenc

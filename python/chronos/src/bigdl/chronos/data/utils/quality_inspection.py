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

import pandas as pd
from bigdl.nano.utils.log4Error import invalidInputError
import logging


def quality_check_timeseries_dataframe(df,
                                       dt_col):
    '''
    detect the low-quality data and provide suggestion (e.g. call .impute or .resample).

    :param df: a pandas dataframe for your raw time series data.
    :param dt_col: a str indicates the col name of datetime
            column in the input data frame, the dt_col must be sorted
            from past to latest respectively for each id.
    '''
    flag = True
    # 1. timestamp check
    flag = flag and _timestamp_type_check(df[dt_col])

    # 2. irregular interval check
    flag = flag and _time_interval_check(df[dt_col])

    # 3. missing value check
    flag = flag and _missing_value_check(df, dt_col)

    # 4. pattern check and noise check
    # TODO:

    return flag


def _timestamp_type_check(df_column):
    '''
    This check is used to make datetime column is datetime64 stype to facilitate our
    access to freq.
    '''
    _is_pd_datetime = pd.api.types.is_datetime64_any_dtype(df_column.dtypes)
    if _is_pd_datetime is not True:
        logging.warning("Datetime colomn should be datetime64 dtype.")
        return False
    return True


def _time_interval_check(df_column):
    '''
    This check is used to verify whether all the time intervals of datetime column
    are consistent.
    '''
    interval = df_column.shift(-1) - df_column
    intervals = interval[:-1].unique()
    if len(intervals) > 1:
        logging.warning("There are irregular interval(more than one interval length)"
                        " among the data, please call .resample(interval).impute() "
                        "first to clean the data.")
        return False
    return True


def _missing_value_check(df, dt_col, threshold=0):
    '''
    This check is used to determine whether there are missing values in the data.
    '''
    for column in df.columns:
        if column == dt_col:
            continue
        df_col = df[column]
        missing_value = df_col.isna().sum()  # na number
        rows = len(df)
        if missing_value / rows > threshold:
            logging.warning(f"The missing value of column {column} exceeds {threshold},"
                            f"please call .impute() fisrt to remove N/A number")
            return False
    return True

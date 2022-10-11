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


def quality_check_timeseries_dataframe(df, dt_col, id_col=None, repair=True):
    '''
    detect the low-quality data and provide suggestion (e.g. call .impute or .resample).

    :param df: a pandas dataframe for your raw time series data.
    :param dt_col: a str indicates the col name of datetime
           column in the input data frame, the dt_col must be sorted
           from past to latest respectively for each id.
    :param id_col: (optional) a str indicates the col name of dataframe id. If
           it is not explicitly stated, then the data is interpreted as only
           containing a single id.
    :param repair: a bool indicates whether automaticly repair low quality data.

    :return: a bool indicates df whether contains low-quality data.
    '''
    invalidInputError(dt_col in df.columns, f"dt_col {dt_col} can not be found in df.")
    invalidInputError(pd.isna(df[dt_col]).sum() == 0, "There is N/A in datetime col")
    flag = True
    # 1. timestamp check
    if _timestamp_type_check(df[dt_col]) is False:
        if repair is True:
            flag = flag and _timestamp_type_repair(df, dt_col)
        else:
            flag = False

    # 2. irregular interval check
    if flag is True:
        interval_flag, interval = _time_interval_check(df[dt_col])
        if interval_flag is False:
            if repair is True:
                df, repair_flag = _time_interval_repair(df, dt_col,
                                                        interval, id_col)
                flag = flag and repair_flag
            else:
                flag = False

    # 3. missing value check
    if _missing_value_check(df, dt_col) is False:
        if repair is True:
            flag = flag and _missing_value_repair(df)
        else:
            flag = False

    # 4. pattern check and noise check
    # TODO:

    return flag, df


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


def _timestamp_type_repair(df, dt_col):
    '''
    This repair is used to convert object or other non datetime64 timestamp column
    to datetime dtype.
    '''
    try:
        df[dt_col].astype('datetime64')
    except:
        return False
    df[dt_col] = df[dt_col].astype('datetime64')
    logging.warning("Datetime colomn has be modified to datetime64 dtype.")
    return True


def _time_interval_check(df_column):
    '''
    This check is used to verify whether all the time intervals of datetime column
    are consistent.
    '''
    interval = df_column.shift(-1) - df_column
    unique_intervals = interval[:-1].unique()
    if len(unique_intervals) > 1:
        logging.warning("There are irregular interval(more than one interval length)"
                        " among the data, please call .resample(interval).impute() "
                        "first to clean the data.")
        return False, interval
    return True, interval


def _time_interval_repair(df, dt_col, interval, id_col=None):
    '''
    This check is used to get consitent time interval by resample data according to
    the mode of original intervals.
    '''
    mode = interval[:-1].mode()[0]  # Timedelta
    from bigdl.chronos.data.utils.resample import resample_timeseries_dataframe
    # TODO: how to change into inplace modification
    try:
        df = resample_timeseries_dataframe(df, dt_col=dt_col,
                                           interval=mode,
                                           id_col=id_col)
        logging.warning(f"Dataframe has be resampled according to interval {mode}.")
        return df, True
    except:
        return df, False


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


def _missing_value_repair(df):
    '''
    This repair is used to fill missing value with impute by linear interpolation.
    '''
    try:
        # interpolate for most cases
        df.interpolate(axis=0, limit_direction='both', inplace=True)
        # fillna with 0 for cases when the whole column is missing
        df.fillna(0, inplace=True)
    except:
        return False
    logging.warning("Missing data has be imputed.")
    return True

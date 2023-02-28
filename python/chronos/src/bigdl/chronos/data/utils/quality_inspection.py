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
import numpy as np
from bigdl.nano.utils.common import invalidInputError
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
    if id_col is not None:
        invalidInputError(id_col in df.columns, f"id_col {id_col} can not be found in df.")
    invalidInputError(pd.isna(df[dt_col]).sum() == 0, "There is N/A in datetime col")
    if df.empty is True:
        return True, df
    flag = True
    # 1. timestamp check
    if _timestamp_type_check(df[dt_col]) is False:
        if repair is True:
            flag = flag and _timestamp_type_repair(df, dt_col)
        else:
            flag = False

    # 2. irregular interval check
    if flag is True:
        interval_flag, intervals = _time_interval_check(df, dt_col, id_col)
        if interval_flag is False:
            if repair is True:
                df, repair_flag = _time_interval_repair(df, dt_col,
                                                        intervals, id_col)
                flag = flag and repair_flag
            else:
                flag = False

    # 3. missing value check
    if _missing_value_check(df, dt_col) is False:
        if repair is True:
            flag = flag and _missing_value_repair(df, dt_col)
        else:
            flag = False

    # 4. pattern check and noise check
    # TODO:

    # 5. abnormal value check
    _abnormal_value_check(df, dt_col)

    return flag, df


def _timestamp_type_check(df_column):
    '''
    This check is used to make datetime column is datetime64 stype to facilitate our
    access to freq.
    '''
    _is_pd_datetime = pd.api.types.is_datetime64_any_dtype(df_column.dtypes)
    if _is_pd_datetime is not True:
        logging.warning("Datetime column should be datetime64 dtype. You can manually modify "
                        "the dtype, or set repair=True when initialize TSDataset.")
        return False
    return True


def _timestamp_type_repair(df, dt_col):
    '''
    This repair is used to convert object or other non datetime64 timestamp column
    to datetime dtype.
    '''
    try:
        df[dt_col] = df[dt_col].astype('datetime64')
    except:
        return False
    logging.warning("Datetime column has be modified to datetime64 dtype.")
    return True


def _time_interval_check(df, dt_col, id_col=None):
    '''
    This check is used to verify whether all the time intervals of datetime column
    are consistent.
    '''
    if id_col is not None:
        # _id_list = list(np.unique(df[id_col]))
        _id_list = df[id_col].unique()
    # check whether exists multi id
    if id_col is not None and len(_id_list) > 1:
        flag = True

        def get_interval(x):
            df_column = x[dt_col]
            interval = df_column.shift(-1) - df_column
            unique_intervals = interval[:-1].unique()
            return unique_intervals
        group = df.groupby(id_col).apply(get_interval)
        for ind in group.index:
            unique_intervals = group[ind]
            if len(unique_intervals) > 1:
                flag = False
        if flag is True:
            return True, None
        else:
            logging.warning("There are irregular interval(more than one interval length)"
                            " among the data. You can call .resample(interval).impute() "
                            "first to clean the data manually, or set repair=True when "
                            "initialize TSDataset.")
            return False, None
    else:
        df_column = df[dt_col]
        intervals = df_column.shift(-1) - df_column
        unique_intervals = intervals[:-1].unique()
        if len(unique_intervals) > 1:
            logging.warning("There are irregular interval(more than one interval length)"
                            " among the data. You can call .resample(interval).impute() "
                            "first to clean the data manually, or set repair=True when "
                            "initialize TSDataset.")
            return False, intervals
        return True, intervals


def _time_interval_repair(df, dt_col, intervals, id_col=None):
    '''
    This check is used to get consitent time interval by resample data according to
    the mode of original intervals.
    '''
    if id_col is not None and intervals is None:
        # multi_id case
        from bigdl.chronos.data.utils.resample import resample_timeseries_dataframe
        try:
            def resample_interval(x):
                df_column = x[dt_col]
                interval = df_column.shift(-1) - df_column
                intervals = interval[:-1]
                mode = intervals.mode()[0]  # Timedelta
                df = resample_timeseries_dataframe(x, dt_col=dt_col,
                                                   interval=mode,
                                                   id_col=id_col)
                return df
            new_df = df.groupby(id_col, as_index=False).apply(resample_interval)
            new_df.reset_index(drop=True, inplace=True)
            logging.warning("Dataframe has be resampled.")
            return new_df, True
        except:
            return df, False
    else:
        mode = intervals[:-1].mode()[0]  # Timedelta
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
                            f"please call .impute() fisrt to remove N/A number manually, "
                            f"or set repair=True when initialize TSDataset.")
            return False
    return True


def _missing_value_repair(df, dt_col):
    '''
    This repair is used to fill missing value with impute by linear interpolation.
    '''
    try:
        # interpolate for most cases
        temp_col = df[dt_col]
        df[dt_col] = 0
        df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
        df[dt_col] = temp_col
        # fillna with 0 for cases when the whole column is missing
        df.fillna(0, inplace=True)
    except:
        return False
    logging.warning("Missing data has be imputed.")
    return True


def _abnormal_value_check(df, dt_col, threshold=10):
    '''
    This check is used to determine whether there are abnormal values in the data.
    '''
    for column in df.columns:
        if column == dt_col or pd.api.types.is_string_dtype(df[column]):
            continue
        df_col = df[column]
        std_val = df_col.std()
        mean_val = df_col.mean()
        df_col = df_col.apply(lambda x: x - mean_val)
        if df_col.max() > std_val * threshold or df_col.min() < -std_val * threshold:
            logging.warning(f"Some values of column {column} exceeds the mean plus/minus "
                            f"{threshold} times standard deviation, please call "
                            f".repair_abnormal_data() to remove abnormal values.")
            return False
    return True


def _abnormal_value_repair(df, dt_col, mode, threshold):
    '''
    This repair is used to replace detected abnormal data with the last non N/A number.
    '''
    invalidInputError(mode in ['absolute', 'relative'],
                      f"mode should be one of ['absolute', 'relative'], but found {mode}.")
    if mode == 'absolute':
        invalidInputError(isinstance(threshold, tuple),
                          "threshold should be a tuple when mode is set to 'absolute', "
                          f"but found {type(threshold)}.")
        invalidInputError(threshold[0] <= threshold[1],
                          "threshold should be a tuple (min_value, max_value) when mode "
                          f"is set to 'absolute', but found {threshold}.")
        res_df = _abs_abnormal_value_repair(df, dt_col, threshold)
    else:
        invalidInputError(isinstance(threshold, float),
                          "threshold should be a float when mode is set to 'relative', "
                          f"but found {type(threshold)}.")
        res_df = _rel_abnormal_value_repair(df, dt_col, threshold)
    return res_df


def _abs_abnormal_value_repair(df, dt_col, threshold):
    res_df = df.copy()
    for column in res_df.columns:
        if column == dt_col or pd.api.types.is_string_dtype(res_df[column]):
            continue
        res_df[column] = res_df[column].apply(lambda x: np.nan if x < threshold[0] or
                                              x > threshold[1] else x)
    res_df.iloc[0] = res_df.iloc[0].fillna(0)
    res_df = res_df.fillna(method='pad')
    return res_df


def _rel_abnormal_value_repair(df, dt_col, threshold):
    res_df = df.copy()
    for column in res_df.columns:
        if column == dt_col or pd.api.types.is_string_dtype(res_df[column]):
            continue
        std_val = res_df[column].std()
        mean_val = res_df[column].mean()
        res_df[column] = res_df[column].apply(lambda x: np.nan
                                              if x > mean_val + threshold * std_val or
                                              x < mean_val - threshold * std_val else x)
    res_df.iloc[0] = res_df.iloc[0].fillna(0)
    res_df = res_df.fillna(method='pad')
    return res_df

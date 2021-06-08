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
import numpy as np
from packaging import version

import tsfresh
from tsfresh import extract_features


def _is_awake(hour):
    return (((hour >= 6) & (hour <= 23)) | (hour == 0)).astype(int).values


def _is_busy_hours(hour):
    return (((hour >= 7) & (hour <= 9)) | (hour >= 16) & (hour <= 19)).astype(int).values


def _is_weekend(weekday):
    return (weekday >= 5).values


TIME_FEATURE = ("MINUTE", "DAY", "DAYOFYEAR", "HOUR", "WEEKDAY", "WEEKOFYEAR", "MONTH")
ADDITIONAL_TIME_FEATURE_HOUR = {"IS_AWAKE": _is_awake,
                                "IS_BUSY_HOURS": _is_busy_hours}
ADDITIONAL_TIME_FEATURE_WEEKDAY = {"IS_WEEKEND": _is_weekend}


def generate_dt_features(input_df, dt_col):
    '''
    generate features related to datetime
    :param input_df: pandas dataframe
    :param dt_col: col name of the datetime in `input_df`
    '''
    df = input_df.copy()
    field = df[dt_col]

    # built in time features
    for attr in TIME_FEATURE:
        if attr == "WEEKOFYEAR" and \
                version.parse(pd.__version__) >= version.parse("1.1.0"):
            field_datetime = pd.to_datetime(field.values.astype(np.int64))
            df[attr + "({})".format(dt_col)] =\
                pd.Int64Index(field_datetime.isocalendar().week)
        else:
            df[attr + "({})".format(dt_col)] = getattr(field.dt, attr.lower())

    # additional time features
    hour = field.dt.hour
    for attr in ADDITIONAL_TIME_FEATURE_HOUR:
        df[attr + "({})".format(dt_col)] = ADDITIONAL_TIME_FEATURE_HOUR[attr](hour)

    weekday = field.dt.weekday
    for attr in ADDITIONAL_TIME_FEATURE_WEEKDAY:
        df[attr + "({})".format(dt_col)] = ADDITIONAL_TIME_FEATURE_WEEKDAY[attr](weekday)

    return df


def generate_global_features(input_df,
                             column_id,
                             column_sort,
                             default_fc_parameters=None,
                             kind_to_fc_parameters=None):
    '''
    generate global features by tsfresh.
    :param input_df: input dataframe.
    :param column_id: id column name
    :param column_sort: time column name
    :param default_fc_parameters: same as tsfresh.
    :param kind_to_fc_parameters: same as tsfresh.

    :return : a new input_df that contains all generated feature.
    '''
    if kind_to_fc_parameters is not None:
        global_feature = extract_features(input_df,
                                          column_id=column_id,
                                          column_sort=column_sort,
                                          kind_to_fc_parameters=kind_to_fc_parameters)
    else:
        global_feature = extract_features(input_df,
                                          column_id=column_id,
                                          column_sort=column_sort,
                                          default_fc_parameters=default_fc_parameters)
    res_df = input_df.copy()
    id_list = list(np.unique(input_df[column_id]))
    addtional_feature = []
    for col_name in global_feature.columns:
        # any feature that can not be extracted will be dropped
        if global_feature[col_name].isna().sum() > 0:
            continue
        # const value will be given to each univariate time series
        for id_name in id_list:
            res_df.loc[input_df["id"] == id_name, col_name] = global_feature.loc[id_name][col_name]
        addtional_feature.append(col_name)
    return res_df, addtional_feature

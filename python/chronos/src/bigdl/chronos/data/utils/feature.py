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
from packaging import version


def _is_awake(hour):
    return (((hour >= 6) & (hour <= 23)) | (hour == 0)).astype(int).values


def _is_busy_hours(hour):
    return (((hour >= 7) & (hour <= 9)) | (hour >= 16) & (hour <= 19)).astype(int).values


def _is_weekend(weekday):
    return (weekday >= 5).values.astype(np.int64)


TIME_FEATURE = ("MINUTE", "DAY", "DAYOFYEAR", "HOUR", "WEEKDAY", "WEEKOFYEAR", "MONTH", "YEAR")
ADDITIONAL_TIME_FEATURE_HOUR = {"IS_AWAKE": _is_awake,
                                "IS_BUSY_HOURS": _is_busy_hours}
ADDITIONAL_TIME_FEATURE_WEEKDAY = {"IS_WEEKEND": _is_weekend}
FEATURE_INTERVAL = {"MINUTE": pd.Timedelta('1m'),
                    "DAY": pd.Timedelta('1D'),
                    "DAYOFYEAR": pd.Timedelta('1D'),
                    "HOUR": pd.Timedelta('1T'),
                    "WEEKDAY": pd.Timedelta('1D'),
                    "WEEKOFYEAR": pd.Timedelta('1W'),
                    "MONTH": pd.Timedelta('30D'),
                    "YEAR": pd.Timedelta('365D'),
                    "IS_AWAKE": pd.Timedelta('1T'),
                    "IS_BUSY_HOURS": pd.Timedelta('1T'),
                    "IS_WEEKEND": pd.Timedelta('1D')}
FEATURE_BIN_NUM = {"MINUTE": range(0, 60),
                   "DAY": range(1, 32),
                   "DAYOFYEAR": range(1, 367),
                   "HOUR": range(0, 24),
                   "WEEKDAY": range(0, 7),
                   "WEEKOFYEAR": range(1, 54),
                   "MONTH": range(1, 13),
                   "YEAR": range(1970, 2099)}


def _one_hot_encode_helper(df, class_name, class_range, features_generated):
    for i in class_range:
        df[class_name + "_" + str(i)] = 0
        df.loc[df[class_name] == i, class_name + "_" + str(i)] = 1
        features_generated.append(class_name + "_" + str(i))
    df.drop([class_name], axis=1, inplace=True)
    features_generated.remove(class_name)
    return df


def generate_dt_features(input_df, dt_col, features, one_hot_features, freq, features_generated):
    '''
    generate features related to datetime
    :param input_df: pandas dataframe
    :param dt_col: col name of the datetime in `input_df`
    :param features: same as the param in TSDataset interface
    :param one_hot_features: same as the param in TSDataset interface
    :param freq: data frequency
    :param features_generated:

    :return : df
    '''
    # get feature generation candidate
    if isinstance(features, list):
        features_normal = set(features)
    if isinstance(features, str):
        if features == "auto":
            features_normal = set([key for key, value in FEATURE_INTERVAL.items() if value >= freq])
        if features == "all":
            features_normal = set([key for key in FEATURE_INTERVAL.keys()])
    if one_hot_features is None:
        one_hot_features = []
    features_onehot = set(one_hot_features)
    features_normal = features_normal - features_onehot

    df = input_df.copy()
    field = df[dt_col]

    # built in time features
    for attr in TIME_FEATURE:
        if attr not in features_onehot and attr not in features_normal:
            continue
        if attr == "WEEKOFYEAR" and \
                version.parse(pd.__version__) >= version.parse("1.1.0"):
            field_datetime = pd.to_datetime(field.values.astype(np.int64))
            df[attr] =\
                pd.Int64Index(field_datetime.isocalendar().week)
        else:
            df[attr] = getattr(field.dt, attr.lower())
        features_generated.append(attr)
        if attr in features_onehot:
            df = _one_hot_encode_helper(df, attr, FEATURE_BIN_NUM[attr], features_generated)

    # additional time features
    hour = field.dt.hour
    for attr in ADDITIONAL_TIME_FEATURE_HOUR:
        if attr not in features_onehot and attr not in features_normal:
            continue
        df[attr] = ADDITIONAL_TIME_FEATURE_HOUR[attr](hour)
        features_generated.append(attr)

    weekday = field.dt.weekday
    for attr in ADDITIONAL_TIME_FEATURE_WEEKDAY:
        if attr not in features_onehot and attr not in features_normal:
            continue
        df[attr] = ADDITIONAL_TIME_FEATURE_WEEKDAY[attr](weekday)
        features_generated.append(attr)

    return df


def generate_global_features(input_df,
                             column_id,
                             column_sort,
                             default_fc_parameters=None,
                             kind_to_fc_parameters=None,
                             n_jobs=1):
    '''
    generate global features by tsfresh.
    :param input_df: input dataframe.
    :param column_id: id column name
    :param column_sort: time column name
    :param default_fc_parameters: same as tsfresh.
    :param kind_to_fc_parameters: same as tsfresh.
    :param n_jobs: int. The number of processes to use for parallelization.

    :return : a new input_df that contains all generated feature.
    '''
    from tsfresh import extract_features
    if kind_to_fc_parameters is not None:
        global_feature = extract_features(input_df,
                                          column_id=column_id,
                                          column_sort=column_sort,
                                          kind_to_fc_parameters=kind_to_fc_parameters,
                                          n_jobs=n_jobs)
    else:
        global_feature = extract_features(input_df,
                                          column_id=column_id,
                                          column_sort=column_sort,
                                          default_fc_parameters=default_fc_parameters,
                                          n_jobs=n_jobs)
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

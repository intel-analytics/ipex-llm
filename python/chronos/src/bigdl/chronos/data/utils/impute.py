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


def impute_timeseries_dataframe(df,
                                dt_col,
                                mode="last",
                                const_num=0):
    '''
    impute and return a dataframe without N/A.
    :param df: input dataframe.
    :param dt_col: name of datetime colomn.
    :param mode: imputation mode, select from "last", "const" or "linear".
           "last": impute by propagating the last non N/A number to its following N/A.
                   if there is no non N/A number ahead, 0 is filled instead.
           "const": impute by a const value input by user.
           "linear": impute by linear interpolation.
    :param const_num: only effective when mode is set to "const".
    '''
    assert dt_col in df.columns, f"dt_col {dt_col} can not be found in df."
    assert pd.isna(df[dt_col]).sum() == 0, "There is N/A in datetime col"
    assert mode in ['last', 'const', 'linear'],\
        f"mode should be one of ['last', 'const', 'linear'], but found {mode}."

    res_df = None
    if mode == "last":
        res_df = _last_impute_timeseries_dataframe(df)
    if mode == "const":
        res_df = _const_impute_timeseries_dataframe(df, const_num)
    if mode == "linear":
        res_df = _linear_impute_timeseries_dataframe(df)

    return res_df


def _last_impute_timeseries_dataframe(df):
    res_df = df.copy()
    res_df.iloc[0] = res_df.iloc[0].fillna(0)
    res_df = res_df.fillna(method='pad')
    return res_df


def _const_impute_timeseries_dataframe(df, const_num):
    res_df = df.fillna(const_num)
    return res_df


def _linear_impute_timeseries_dataframe(df):
    res_df = df.interpolate(axis=0, limit_direction='both')
    return res_df

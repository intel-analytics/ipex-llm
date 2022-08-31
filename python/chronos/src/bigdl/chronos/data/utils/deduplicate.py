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


def deduplicate_timeseries_dataframe(df,
                                     dt_col):
    '''
    deduplicate and return a dataframe with no identical rows.
    :param df: input dataframe.
    :param dt_col: name of datetime colomn.
    '''
    from bigdl.nano.utils.log4Error import invalidInputError
    invalidInputError(dt_col in df.columns, f"dt_col {dt_col} can not be found in df.")
    invalidInputError(pd.isna(df[dt_col]).sum() == 0, "There is N/A in datetime col")
    res_df = df.drop_duplicates()
    return res_df

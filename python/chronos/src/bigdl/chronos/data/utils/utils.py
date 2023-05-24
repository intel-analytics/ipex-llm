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


def _to_list(item, name, expect_type=str, deploy_mode=False):
    if isinstance(item, list):
        return item
    if item is None:
        return []
    if not deploy_mode:
        _check_type(item, name, expect_type)
    return [item]


def _check_type(item, name, expect_type):
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(isinstance(item, expect_type),
                      f"a {str(expect_type)} is expected for {name} but found {type(item)}")


def _check_col_within(df, col_name):
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(col_name in df.columns,
                      f"{col_name} is expected in dataframe while not found")


def _check_col_no_na(df, col_name):
    from bigdl.nano.utils.common import invalidInputError
    _check_col_within(df, col_name)
    invalidInputError(df[col_name].isna().sum() == 0,
                      f"{col_name} column should not have N/A.")


def _check_cols_no_na(df, col_names):
    col_names = _to_list(col_names, name=None)
    for col_name in col_names:
        _check_col_no_na(df, col_name)


def _check_is_aligned(df, id_col, dt_col):
    res = len(set(df.groupby(id_col).apply(lambda df: hash(str(df[dt_col].values))))) == 1
    return res


def _check_dt_is_sorted(df, dt_col):
    import numpy as np
    import warnings
    df = df.copy()
    try:
        res = (np.diff(df[dt_col].values.astype(np.float32)) >= 0).all()
        if not res:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              f"{dt_col} must be sorted.")
    except (ValueError, TypeError):
        warnings.warn(f"{dt_col} may not be sorted.", Warning)

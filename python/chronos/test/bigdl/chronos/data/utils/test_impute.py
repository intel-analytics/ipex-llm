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

import pytest
import pandas as pd
import numpy as np

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.data.utils.impute import impute_timeseries_dataframe, \
    _last_impute_timeseries_dataframe, _const_impute_timeseries_dataframe, \
    _linear_impute_timeseries_dataframe


def get_ugly_ts_df():
    data = np.random.random_sample((50, 5))
    mask = np.random.random_sample((50, 5))
    mask[mask >= 0.4] = 2
    mask[mask < 0.4] = 1
    mask[mask < 0.2] = 0
    data[mask == 0] = None
    data[mask == 1] = np.nan
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'][0] = np.nan  # make sure column 'a' has a N/A
    df["datetime"] = pd.date_range('1/1/2019', periods=50)
    return df


class TestImputeTimeSeries(ZooTestCase):
    def setup_method(self, method):
        self.df = get_ugly_ts_df()

    def teardown_method(self, method):
        pass

    def test_impute_timeseries_dataframe(self):
        with pytest.raises(AssertionError):
            impute_timeseries_dataframe(self.df, dt_col="z")
        with pytest.raises(AssertionError):
            impute_timeseries_dataframe(
                self.df, dt_col="datetime", mode="dummy")
        with pytest.raises(AssertionError):
            impute_timeseries_dataframe(self.df, dt_col="a")
        last_res_df = impute_timeseries_dataframe(
            self.df, dt_col="datetime", mode="last")
        assert self.df.isna().sum().sum() != 0
        assert last_res_df.isna().sum().sum() == 0
        const_res_df = impute_timeseries_dataframe(
            self.df, dt_col="datetime", mode="const")
        assert self.df.isna().sum().sum() != 0
        assert const_res_df.isna().sum().sum() == 0
        linear_res_df = impute_timeseries_dataframe(
            self.df, dt_col="datetime", mode="linear")
        assert self.df.isna().sum().sum() != 0
        assert linear_res_df.isna().sum().sum() == 0

    def test_last_impute_timeseries_dataframe(self):
        data = {'data': [np.nan, np.nan, 1, np.nan, 2, 3]}
        df = pd.DataFrame(data)
        res_df = _last_impute_timeseries_dataframe(df)
        assert res_df['data'][0] == 0
        assert res_df['data'][1] == 0
        assert res_df['data'][3] == 1

    def test_const_impute_timeseries_dataframe(self):
        data = {'data': [np.nan, 1, np.nan, 2, 3]}
        df = pd.DataFrame(data)
        res_df = _const_impute_timeseries_dataframe(df, 1)
        assert res_df['data'][0] == 1
        assert res_df['data'][2] == 1

    def test_linear_timeseries_dataframe(self):
        data = {'data': [np.nan, 1, np.nan, 2, 3]}
        df = pd.DataFrame(data)
        res_df = _linear_impute_timeseries_dataframe(df)
        assert res_df['data'][0] == 1
        assert res_df['data'][2] == 1.5

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

from unittest import TestCase
from bigdl.chronos.data.utils.quality_inspection import quality_check_timeseries_dataframe
from ... import op_torch, op_tf2


def get_missing_df():
    data = np.random.random_sample((50, 5))
    mask = np.random.random_sample((50, 5))
    mask[mask >= 0.4] = 2
    mask[mask < 0.4] = 1
    mask[mask < 0.2] = 0
    data[mask == 0] = None
    data[mask == 1] = np.nan
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'] = np.nan  # make sure column 'a' has are all N/A
    df["datetime"] = pd.date_range('1/1/2019', periods=50)
    return df


def get_multi_interval_df():
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df["datetime"] = pd.date_range('1/1/2019', periods=50)
    # modify datetime to generate a different interval
    df["datetime"][25:] = pd.date_range('1/1/2020', periods=25)
    return df


def get_non_dt_df():
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df["datetime"] = ["2022-1-1"]*50
    return df


@op_torch
@op_tf2
class TestCheckAndRepairTimeSeries(TestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass
    
    def test_normal_dataframe(self):
        pass

    def test_missing_check_and_repair(self):
        df = get_missing_df()
        flag, _ = quality_check_timeseries_dataframe(df, "datetime", repair=False)
        assert flag is False
        flag, _ = quality_check_timeseries_dataframe(df, "datetime", repair=True)
        assert flag is True
        # make sure modifiation has been made to df
        flag, _ = quality_check_timeseries_dataframe(df, "datetime", repair=False)
        assert flag is True

    def test_time_interval_check_and_repair(self):
        df = get_multi_interval_df()
        flag, _ = quality_check_timeseries_dataframe(df, "datetime", repair=False)
        assert flag is False
        flag, df = quality_check_timeseries_dataframe(df, "datetime", repair=True)
        assert flag is True
        # make sure modifiation has been made to df
        flag, _ = quality_check_timeseries_dataframe(df, "datetime", repair=False)
        assert flag is True

    def test_non_dt_type_check_and_repair(self):
        df = get_non_dt_df()
        flag, _ = quality_check_timeseries_dataframe(df, "datetime", repair=False)
        assert flag is False
        flag, df = quality_check_timeseries_dataframe(df, "datetime", repair=True)
        assert flag is True
        # make sure modifiation has been made to df
        flag, _ = quality_check_timeseries_dataframe(df, "datetime", repair=False)
        assert flag is True

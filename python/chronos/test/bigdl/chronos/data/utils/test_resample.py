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
from bigdl.chronos.data.utils.resample import resample_timeseries_dataframe


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


class TestResampleTimeSeries(ZooTestCase):
    def setup_method(self, method):
        self.df = get_ugly_ts_df()

    def teardown_method(self, method):
        pass

    def test_resample_timeseries_dataframe(self):
        with pytest.raises(AssertionError):
            resample_timeseries_dataframe(self.df, dt_col="z",
                                          interval="1D",
                                          start_time='1/1/2019',
                                          end_time='1/5/2019',
                                          merge_mode='max')
        with pytest.raises(AssertionError):
            resample_timeseries_dataframe(self.df, dt_col="z",
                                          interval="1D",
                                          start_time='1/1/2019',
                                          end_time='1/5/2019',
                                          merge_mode='dummy')
        with pytest.raises(AssertionError):
            resample_timeseries_dataframe(self.df, dt_col="a",
                                          interval="1D",
                                          start_time='1/1/2019',
                                          end_time='1/5/2019',
                                          merge_mode='max')
        with pytest.raises(AssertionError):
            resample_timeseries_dataframe(self.df, dt_col="datetime",
                                          interval="1D",
                                          start_time='1/10/2019',
                                          end_time='1/5/2019',
                                          merge_mode='max')
        res_df = resample_timeseries_dataframe(self.df, dt_col="datetime",
                                               interval="3D",
                                               start_time='12/20/2018',
                                               end_time='1/5/2019',
                                               merge_mode='max')
        assert len(res_df) == 6

    def test_resample_timeseries_dataframe_modes(self):
        data = {
            'data': [
                1, 2, 3], 'datetime': [
                "2020-11-09T08", "2020-11-09T09", "2020-11-09T11"]}
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2H",
            start_time='2020-11-09T07',
            end_time='2020-11-09T10',
            merge_mode='max')
        assert np.isnan(
            res_df['data'][0]) and res_df['data'][1] == 2 and res_df['data'][2] == 3
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2H",
            start_time='2020-11-09T07',
            end_time='2020-11-09T10',
            merge_mode='min')
        assert np.isnan(
            res_df['data'][0]) and res_df['data'][1] == 1 and res_df['data'][2] == 3
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2H",
            start_time='2020-11-09T07',
            end_time='2020-11-09T10',
            merge_mode='mean')
        assert np.isnan(
            res_df['data'][0]) and res_df['data'][1] == 1.5 and res_df['data'][2] == 3
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2H",
            start_time='2020-11-09T07',
            end_time='2020-11-09T10',
            merge_mode='sum')
        assert np.isnan(
            res_df['data'][0]) and res_df['data'][1] == 3 and res_df['data'][2] == 3

    def test_resample_timeseries_dataframe_ms(self):
        data = {
            'data': [
                1,
                2,
                3],
            'datetime': [
                "2020-11-09T07:52:00.007",
                "2020-11-09T07:52:00.008",
                "2020-11-09T07:52:00.010"]}
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2ms",
            start_time='2020-11-09T07:52:00.005',
            end_time='2020-11-09T07:52:00.010',
            merge_mode='max')
        assert len(res_df) == 4 and res_df['data'].isna().sum() == 1
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2ms",
            start_time='2020-11-09T07:52:00.006',
            end_time='2020-11-09T07:52:00.010',
            merge_mode='max')
        assert len(res_df) == 3 and res_df['data'].isna().sum() == 0
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2ms",
            start_time='2020-11-09T07:52:00.007',
            end_time='2020-11-09T07:52:00.010',
            merge_mode='max')
        assert len(res_df) == 3 and res_df['data'].isna().sum() == 0
        res_df = resample_timeseries_dataframe(
            df,
            dt_col="datetime",
            interval="2ms",
            merge_mode='max')
        assert len(res_df) == 3 and res_df['data'].isna().sum() == 0

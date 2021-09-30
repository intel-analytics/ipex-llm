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
from bigdl.chronos.data.utils.deduplicate import deduplicate_timeseries_dataframe


def get_duplicated_ugly_ts_df():
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'][0] = np.nan  # make sure column 'a' has a N/A
    df["datetime"] = pd.date_range('1/1/2019', periods=50)
    for i in range(20):
        df.loc[len(df)] = df.loc[np.random.randint(0, 49)]
    return df


class TestDeduplicateTimeSeries(ZooTestCase):
    def setup_method(self, method):
        self.df = get_duplicated_ugly_ts_df()

    def teardown_method(self, method):
        pass

    def test_deduplicate_timeseries_dataframe(self):
        with pytest.raises(AssertionError):
            deduplicate_timeseries_dataframe(self.df, dt_col="z")
        with pytest.raises(AssertionError):
            deduplicate_timeseries_dataframe(self.df, dt_col="a")
        res_df = deduplicate_timeseries_dataframe(
            self.df, dt_col="datetime")
        assert len(res_df) == 50

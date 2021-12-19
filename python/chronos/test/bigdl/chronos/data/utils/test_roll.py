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
import random

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.data.utils.roll import roll_timeseries_dataframe


class TestRollTimeSeries(ZooTestCase):
    def setup_method(self, method):
        self.easy_data = pd.DataFrame({"A": np.random.randn(10),
                                       "B": np.random.randn(10),
                                       "C": np.random.randn(10),
                                       "datetime": pd.date_range('1/1/2019', periods=10)})
        self.lookback = random.randint(1, 5)

    def teardown_method(self, method):
        pass

    def test_roll_timeseries_dataframe(self):
        x, y = roll_timeseries_dataframe(self.easy_data,
                                         None,
                                         lookback=self.lookback,
                                         horizon=[1, 3],
                                         feature_col=["A"],
                                         target_col=["B"])
        assert x.shape == (8-self.lookback, self.lookback, 2)
        assert y.shape == (8-self.lookback, 2, 1)

        x, y = roll_timeseries_dataframe(self.easy_data,
                                         None,
                                         lookback=self.lookback,
                                         horizon=4,
                                         feature_col=["A", "C"],
                                         target_col=["B"])
        assert x.shape == (7-self.lookback, self.lookback, 3)
        assert y.shape == (7-self.lookback, 4, 1)

        x, y = roll_timeseries_dataframe(self.easy_data,
                                         None,
                                         lookback=2,
                                         horizon=0,
                                         feature_col=[],
                                         target_col=["A"])
        assert x.shape == (9, 2, 1)
        assert y is None

        self.easy_data["A"][0] = None
        x, y = roll_timeseries_dataframe(self.easy_data,
                                         None,
                                         lookback=2,
                                         horizon=0,
                                         feature_col=[],
                                         target_col=["A"])
        assert x.shape == (8, 2, 1)
        assert y is None

        x, y = roll_timeseries_dataframe(self.easy_data,
                                         None,
                                         lookback=2,
                                         horizon=2,
                                         feature_col=["C"],
                                         target_col=["A"])
        assert x.shape == (6, 2, 2)
        assert y.shape == (6, 2, 1)

        x, y = roll_timeseries_dataframe(self.easy_data,
                                         None,
                                         lookback=2,
                                         horizon=2,
                                         feature_col=["A"],
                                         target_col=["B", "C"])
        assert x.shape == (6, 2, 3)
        assert y.shape == (6, 2, 2)

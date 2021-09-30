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

import numpy as np
import pandas as pd
import pytest

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.autots.deprecated.feature.time_sequence import TimeSequenceFeatureTransformer
from bigdl.chronos.autots.deprecated.preprocessing.impute.LastFill import LastFill
from bigdl.chronos.autots.deprecated.preprocessing.impute import FillZeroImpute


class TestDataImputation(ZooTestCase):

    def setup_method(self, method):
        self.ft = TimeSequenceFeatureTransformer()
        self.create_data()

    def teardown_method(self, method):
        pass

    def create_data(self):
        data = np.random.random_sample((5, 50))
        mask = np.random.random_sample((5, 50))
        mask[mask >= 0.4] = 2
        mask[mask < 0.4] = 1
        mask[mask < 0.2] = 0
        data[mask == 0] = None
        data[mask == 1] = np.nan
        df = pd.DataFrame(data)
        idx = pd.date_range(
            start='2020-07-01 00:00:00',
            end='2020-07-01 08:00:00',
            freq='2H')
        df.index = idx
        self.data = df

    def test_lastfill(self):
        last_fill = LastFill()
        mse_missing = last_fill.evaluate(self.data, 0.1)
        imputed_data = last_fill.impute(self.data)
        assert imputed_data.isna().sum().sum() == 0
        mse = last_fill.evaluate(imputed_data, 0.1)

    def test_fillzero(self):
        data = [[np.nan, 1], [np.nan, 2]]
        df = pd.DataFrame(data)
        imputor = FillZeroImpute()
        imputed_df = imputor.impute(df)
        assert df.isna().sum().sum() != 0
        assert imputed_df.isna().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__])

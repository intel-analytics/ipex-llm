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

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.autots.deprecated.preprocessing.utils import train_val_test_split
import pandas as pd
import numpy as np


class TestUtil(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_train_val_test_split(self):
        # length test
        sample_num = 100
        look_back = 10
        horizon = 1
        dates = pd.date_range('1/1/2020', periods=sample_num)
        values = np.random.randn(sample_num)
        df = pd.DataFrame({"values": values}, index=dates)
        train_df, val_df, test_df = train_val_test_split(df,
                                                         val_ratio=0.1,
                                                         test_ratio=0.1,
                                                         look_back=look_back,
                                                         horizon=horizon)
        assert len(train_df) == sample_num * 0.8
        assert len(val_df) == sample_num * 0.1 + look_back + horizon - 1
        assert len(test_df) == sample_num * 0.1 + look_back + horizon - 1
        # index test
        assert pd.api.types.is_datetime64_any_dtype(test_df.index.dtype)
        assert pd.api.types.is_datetime64_any_dtype(val_df.index.dtype)

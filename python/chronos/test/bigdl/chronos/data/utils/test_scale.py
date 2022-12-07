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
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.data.utils.scale import _standard_scaler_scale_timeseries_numpy
from numpy.testing import assert_array_almost_equal

from ... import op_torch, op_tf2, op_diff_set_all


def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value": np.random.randn(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


class TestScaleNumpy(TestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @op_torch
    @op_tf2
    @op_diff_set_all
    def test_unscale_timeseries_numpy(self):
        df = get_ts_df()
        scaler = StandardScaler()
        col_list = ["value", "extra feature"]
        scaler.fit(df[col_list])
        scaled_data_scaler = scaler.transform(df[col_list])
        scaled_data_numpy = \
            _standard_scaler_scale_timeseries_numpy(df[col_list].values, scaler)
        assert_array_almost_equal(scaled_data_scaler, scaled_data_numpy)

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
from bigdl.chronos.data.utils.cycle_detection import cycle_length_est

from ... import op_torch, op_tf2, op_diff_set_all

class TestCycleDetectionTimeSeries(TestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @op_torch
    @op_tf2
    @op_diff_set_all
    def test_cycle_detection_timeseries_numpy(self):
        data = np.random.randn(100)
        cycle_length = cycle_length_est(data)
        assert 1 <= cycle_length <= 100

#
# Copyright 2021 The BigDL Authors.
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

import unittest
import numpy as np

from bigdl.ppml import FLServer
from bigdl.ppml.algorithms.fgboost_regression import FGBoostRegression
from bigdl.ppml.utils import init_fl_context


class TestHflLogisticRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.fl_server = FLServer()
        self.fl_server.build()
        self.fl_server.start()
        init_fl_context()

    def tearDown(self) -> None:
        self.fl_server.stop()

    def test_dummy_data(self):
        x, y = np.ones([2, 3]), np.ones([2])


if __name__ == '__main__':
    unittest.main()

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

import unittest
import pandas as pd
from bigdl.ppml.fl.data_utils import *


class TestDataUtils(unittest.TestCase):
    def test_pandas_api(self):
        df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        array, _ = convert_to_jtensor(df, feature_columns=["f1"])
        self.assert_(isinstance(array, JTensor))
        self.assertEqual(array.storage.shape, (2, 1))

    def test_numpy_api(self):
        array = np.array([[1, 2], [3, 4]])
        array, _ = convert_to_jtensor(array)
        self.assert_(isinstance(array, JTensor))
        self.assertEqual(array.storage.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()

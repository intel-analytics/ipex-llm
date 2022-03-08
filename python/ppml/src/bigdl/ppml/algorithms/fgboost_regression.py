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

from bigdl.dllib.utils.common import JavaValue
from bigdl.ppml.data_utils import *

from bigdl.ppml import *


class FGBoostRegression(JavaValue):
    def __init__(self, jvalue=None, learning_rate:float=1e-3, max_depth=6, min_child_size=1):
        self.bigdl_type = "float"
        super().__init__(jvalue, self.bigdl_type, learning_rate, max_depth, min_child_size)

    def fit(self, x, y, num_round=5, **kargs):
        x, y = convert_to_jtensor(x, y, **kargs)
        return callBigDlFunc(self.bigdl_type, "fgBoostFit", self.value, x, y, num_round)

    def evaluate(self, x, y, **kargs):
        x, y = convert_to_jtensor(x, y, **kargs)
        return callBigDlFunc(self.bigdl_type, "fgBoostEvaluate", self.value, x, y)

    def predict(self, x, **kargs):
        x, _ = convert_to_jtensor(x, **kargs)
        return callBigDlFunc(self.bigdl_type, "fgBoostPredict", self.value, x).to_ndarray()

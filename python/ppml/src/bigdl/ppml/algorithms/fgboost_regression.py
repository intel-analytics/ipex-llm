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


class FGBoostRegression(JavaValue):
    def __init__(self, jvalue, *args):
        self.bigdl_type = "float"
        super(JavaValue, self).__init__(jvalue, bigdl_type, *args)

    def fit(self, x, y, num_round, feature_cols="features", label_cols="label"):
        x = convert_to_numpy(x, feature_cols)
        y = convert_to_numpy(y, label_cols)
        callBigDlFunc(self.bigdl_type, "fgBoostFit", x, y, num_round)

    def evaluate(self, x, y):
        x = convert_to_numpy(x)
        y = convert_to_numpy(y)
        callBigDlFunc(self.bigdl_type, "fgBoostEvaluate", x, y)

    def predict(self, x):
        x = convert_to_numpy(x)
        callBigDlFunc(self.bigdl_type, "fgBoostPredict", x)

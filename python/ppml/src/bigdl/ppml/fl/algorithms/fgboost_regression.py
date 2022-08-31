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


from time import time
from bigdl.dllib.utils.common import JavaValue
from bigdl.ppml.fl.data_utils import *

from bigdl.ppml.fl import *
from bigdl.ppml.fl.fgboost.utils import add_data
import logging


class FGBoostRegression(FLClientClosable):
    def __init__(self, jvalue=None, learning_rate:float=0.1,
                 max_depth=7, min_child_size=1, server_model_path=None):
        self.bigdl_type = "float"
        super().__init__(jvalue, self.bigdl_type, learning_rate, 
                         max_depth, min_child_size, server_model_path)

    def fit(self, x, y=None, num_round=5, **kargs):
        x = convert_to_numpy(x)
        y = convert_to_numpy(y) if y is not None else None
        add_data(x, self.value, "fgBoostFitAdd", self.bigdl_type)
        ts = time()
        x, y = convert_to_jtensor(x, y, **kargs)
        te = time()
        logging.info(f"ndarray to jtensor: [{te-ts} s]")
        return callBigDlFunc(self.bigdl_type, "fgBoostFitCall", self.value, y, num_round)

    def evaluate(self, x, y=None, batchsize=4, **kargs):
        x, y = convert_to_jtensor(x, y, **kargs)
        return callBigDlFunc(self.bigdl_type, "fgBoostEvaluate", self.value, x, y)

    def predict(self, x, batchsize=4, **kargs):
        i = 0
        result = []
        while i + batchsize < len(x):
            x_batch = x[i:i+batchsize]
            x_batch, _ = convert_to_jtensor(x_batch, **kargs)
            result_batch = callBigDlFunc(self.bigdl_type, "fgBoostPredict", self.value, x_batch).to_ndarray()
            result.append(result_batch.flatten())
            i += batchsize
        x_batch = x[i:]
        x_batch, _ = convert_to_jtensor(x_batch, **kargs)
        result_batch = callBigDlFunc(self.bigdl_type, "fgBoostPredict", self.value, x_batch).to_ndarray()
        result.append(result_batch.flatten())
        flat_result = [x for xs in result for x in xs]
        return np.array(flat_result)

    def save_model(self, dest):
        callBigDlFunc(self.bigdl_type, "fgBoostRegressionSave", self.value, dest)

    @classmethod
    def load_model(cls, src):
        # the jvalue exists here so JVM constructor would not be called again
        # thus the parameters would remain the same as model loaded
        return cls(jvalue=callBigDlFunc("float", "fgBoostRegressionLoad", src))

    def load_server_model(self, model_path):
        callBigDlFunc(self.bigdl_type, "fgBoostLoadServerModel", self.value, model_path)

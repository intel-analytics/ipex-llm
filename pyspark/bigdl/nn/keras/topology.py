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

from bigdl.nn.keras.layer import KerasLayer
from bigdl.util.common import *
import multiprocessing


class KerasModel(KerasLayer):
    def compile(self, optimizer, loss, metrics=None):
        callBigDlFunc(self.bigdl_type, "compile",
                      self.value,
                      optimizer,
                      loss,
                      metrics)

    def fit(self, x, y=None, batch_size=32, nb_epoch=10, validation_data=None, distributed=True):
        init_engine()
        redire_spark_logs()
        show_bigdl_info_logs()
        if distributed:
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                training_data = to_sample_rdd(x, y)
                if validation_data:
                    validation_data = to_sample_rdd(*validation_data)
            elif isinstance(x, RDD) and not y:
                training_data = x
            else:
                raise TypeError("Unsupported training data type: %s" % type(x))
            callBigDlFunc(self.bigdl_type, "fit",
                          self.value,
                          training_data,
                          batch_size,
                          nb_epoch,
                          validation_data)
        else:
            if validation_data:
                val_x = [JTensor.from_ndarray(x) for x in to_list(validation_data[0])]
                val_y = JTensor.from_ndarray(validation_data[1])
            else:
                val_x, val_y = None, None
            callBigDlFunc(self.bigdl_type, "fit",
                          self.value,
                          [JTensor.from_ndarray(x) for x in to_list(x)],
                          JTensor.from_ndarray(y),
                          batch_size,
                          nb_epoch,
                          val_x,
                          val_y,
                          multiprocessing.cpu_count())

    def evaluate(self, x, y=None, batch_size=32):
        init_engine()
        redire_spark_logs()
        show_bigdl_info_logs()
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            evaluation_data = to_sample_rdd(x, y)
        elif isinstance(x, RDD) and not y:
            evaluation_data = x
        else:
            raise TypeError("Unsupported evaluation data type: %s" % type(x))
        return callBigDlFunc(self.bigdl_type, "evaluate",
                             self.value,
                             evaluation_data,
                             batch_size)

    def predict(self, x, distributed=True):
        init_engine()
        redire_spark_logs()
        show_bigdl_info_logs()
        if is_distributed:
            if isinstance(x, np.ndarray):
                features = to_sample_rdd(x, np.zeros([x.shape[0]]))
            elif isinstance(x, RDD):
                features = x
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))
            return self.predict_distributed(features)
        else:
            if isinstance(x, np.ndarray):
                return self.predict_local(x)
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))


class Sequential(KerasModel):
    """
    Container for a Sequential model.

    >>> sequential = Sequential()
    creating: createKerasSequential
    """
    def __init__(self, bigdl_type="float"):
        super(Sequential, self).__init__(None, bigdl_type=bigdl_type)

    def add(self, model):
        self.value.add(model.value)
        return self


class Model(KerasModel):
    def __init__(self, input, output, bigdl_type="float"):
        super(Model, self).__init__(None, bigdl_type,
                                    to_list(input),
                                    to_list(output))
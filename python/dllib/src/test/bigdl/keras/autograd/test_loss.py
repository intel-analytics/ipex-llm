#
# Copyright 2018 Analytics Zoo Authors.
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

import keras.backend as KK
import keras.layers as klayers
import keras.objectives as kloss
import numpy as np
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.autograd import *

np.random.seed(1337)  # for reproducibility

class TestLoss(ZooTestCase):
    # kkloss is a function which accept y_pred and y_true.
    # y_pred and y_true are all keras_tensor
    # zloss is a AbstractCriterion
    # The first dim of shape is batch
    def compareLossWithKeras(self, kkloss_func, zloss, shape):

        y_pred = klayers.Input(shape=shape[1:])
        y_true = klayers.Input(shape=shape[1:])

        batch = shape[0]

        kkloss = kkloss_func(y_true, y_pred)
        y_true_value = np.random.uniform(0, 1, shape)
        y_pred_value = np.random.uniform(0, 1, shape)

        k_grad_y_pred = KK.get_session().run(KK.gradients(kkloss, y_pred),
                                         feed_dict={y_true: y_true_value, y_pred: y_pred_value})[0]
        k_output = KK.get_session().run(kkloss,
                                         feed_dict={y_true: y_true_value, y_pred: y_pred_value})
        z_output = zloss.forward(y_true_value, y_pred_value)
        z_grad_y_pred = zloss.backward(y_true_value, y_pred_value)

        assert(z_output == pytest.approx(np.mean(k_output), 1e-5, 1e-5))
        self.assert_allclose(k_grad_y_pred / batch, z_grad_y_pred)

    def test_abs(self):
        def mean_absolute_error(y_true, y_pred):
            result = mean(abs(y_true - y_pred), axis=1)
            return result
        self.compareLossWithKeras(kloss.mean_absolute_error,
                                  CustomLoss(mean_absolute_error), [2, 3])

if __name__ == "__main__":
   pytest.main([__file__])

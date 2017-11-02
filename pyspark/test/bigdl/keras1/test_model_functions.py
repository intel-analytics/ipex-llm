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
from __future__ import print_function

from keras.layers import *

np.random.seed(1337)  # for reproducibility
import numpy as np
import pytest

import bigdl.keras1.backend as bigdl_backend
import bigdl.util.common as bigdl_common

np.random.seed(1337)  # for reproducibility
from test.bigdl.test_utils import BigDLTestCase, TestModels


# TODO: add method to test optimizers
class TestModelFunctions(BigDLTestCase):

    def test_fit_seq_lenet_mnist(self):
        kmodel, input_data, output_data = TestModels.kmodel_seq_lenet_mnist()
        bigdl_common.init_engine()
        bmodel = bigdl_backend.use_bigdl_backend(kmodel)
        bmodel.fit(input_data, output_data, batch_size=4, nb_epoch=2,
                   validation_data=(input_data, output_data))

    # TODO: enable this
    # def test_fit_graph_linear_regression(self):
    #     input1 = Input(shape=(3,))
    #     dense = Dense(1, bias=False)(input1)
    #     model = Model(input=input1, output=dense)
    #     model.compile(loss='mse',
    #                   optimizer='sgd')
    #     # lr should be 0.01 otherwise the result is bad.
    #     # We cannot set lr, so we loose the toralence to 1e-1 for now.
    #     bigdl_common.init_engine()
    #     bmodel = bigdl_backend.use_bigdl_backend(model)
    #     input_data = np.random.uniform(0, 1, (1000, 3))
    #     expected_W = np.array([1, 2, 3]).transpose()
    #     output_data = np.dot(input_data, expected_W)
    #     bmodel.fit(input_data, output_data, batch_size=4, nb_epoch=10)
    #     actual_W = bmodel.get_weights()
    #     assert_allclose(expected_W, actual_W[0][0], rtol=1e-1)

    def test_evaluate(self):
        kmodel, input_data, output_data = TestModels.kmodel_seq_lenet_mnist()
        bigdl_common.init_engine()
        bmodel = bigdl_backend.use_bigdl_backend(kmodel)
        bmodel.fit(input_data, output_data, batch_size=4, nb_epoch=2,
                   validation_data=(input_data, output_data))
        bmodel.evaluate(input_data, output_data)

if __name__ == "__main__":
    pytest.main([__file__])

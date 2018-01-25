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

import numpy as np
import pytest
from keras import objectives

from bigdl.keras.optimization import OptimConverter
from test.bigdl.test_utils import BigDLTestCase
np.random.seed(1337)  # for reproducibility


class TestLoss(BigDLTestCase):

    def test_mse(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.mean_squared_error
        bloss = OptimConverter.to_bigdl_criterion(objectives.mean_squared_error)
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_msle(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.mean_squared_logarithmic_error
        bloss = OptimConverter.to_bigdl_criterion("msle")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_mae(self):
        y_a = np.random.random([5, 6, 7])
        y_b = np.random.random([5, 6, 7])
        kloss = objectives.mean_absolute_error
        bloss = OptimConverter.to_bigdl_criterion("mean_absolute_error")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_mape(self):
        y_a = np.random.random([5, 6, 7])
        y_b = np.random.random([5, 6, 7])
        kloss = objectives.mean_absolute_percentage_error
        bloss = OptimConverter.to_bigdl_criterion("mape")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_binary_crossentropy(self):
        y_a = np.random.random([5, 6, 7])
        y_b = np.random.random([5, 6, 7])
        kloss = objectives.binary_crossentropy
        bloss = OptimConverter.to_bigdl_criterion("binary_crossentropy")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_categorical_crossentropy(self):
        y_a = np.random.random([2, 3])
        y_b = np.array([[0, 1, 0], [0, 0, 1]])
        kloss = objectives.categorical_crossentropy
        bloss = OptimConverter.to_bigdl_criterion("categorical_crossentropy")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_sparse_categorical_crossentropy(self):
        import keras.backend as K
        y_a = np.array([0.12, 0.22, 0.30, 0.17, 0.19])
        # index starts from 1 in BigDL but starts from 0 in Keras
        y_b_bigdl = np.array([2])
        y_b_keras = np.array([1])
        kloss = objectives.sparse_categorical_crossentropy
        bloss = OptimConverter.to_bigdl_criterion("sparse_categorical_crossentropy")
        bigdl_output = bloss.forward(y_a, y_b_bigdl)
        keras_output = np.mean(K.eval(kloss(K.variable(y_b_keras), K.variable(y_a))))
        np.testing.assert_allclose(bigdl_output, keras_output)

    def test_hinge(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.hinge
        bloss = OptimConverter.to_bigdl_criterion("hinge")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_squared_hinge(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.squared_hinge
        bloss = OptimConverter.to_bigdl_criterion("squared_hinge")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_poisson(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.poisson
        bloss = OptimConverter.to_bigdl_criterion("poisson")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_cosine_proximity(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.cosine_proximity
        bloss = OptimConverter.to_bigdl_criterion("cosine_proximity")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_kld(self):
        y_a = np.random.random([4, 5])
        y_b = np.random.random([4, 5])
        kloss = objectives.kullback_leibler_divergence
        bloss = OptimConverter.to_bigdl_criterion("kld")
        self.compare_loss(y_a, y_b, kloss, bloss)


if __name__ == "__main__":
    pytest.main([__file__])

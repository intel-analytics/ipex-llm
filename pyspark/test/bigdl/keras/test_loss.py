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

    def test_mae(self):
        y_a = np.random.random([5, 6, 7])
        y_b = np.random.random([5, 6, 7])
        kloss = objectives.mean_absolute_error
        bloss = OptimConverter.to_bigdl_criterion("mean_absolute_error")
        self.compare_loss(y_a, y_b, kloss, bloss, rtol=1e-6, atol=1e-6)

    def test_binary_crossentropy(self):
        y_a = np.random.random([5, 6, 7])
        y_b = np.random.random([5, 6, 7])
        kloss = objectives.binary_crossentropy
        bloss = OptimConverter.to_bigdl_criterion("binary_crossentropy")
        self.compare_loss(y_a, y_b, kloss, bloss)

    def test_hinge(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.hinge
        bloss = OptimConverter.to_bigdl_criterion("hinge")
        self.compare_loss(y_a, y_b, kloss, bloss, rtol=1e-6, atol=1e-6)

    def test_squared_hinge(self):
        y_a = np.random.random([2, 3, 4])
        y_b = np.random.random([2, 3, 4])
        kloss = objectives.squared_hinge
        bloss = OptimConverter.to_bigdl_criterion("squared_hinge")
        self.compare_loss(y_a, y_b, kloss, bloss, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])

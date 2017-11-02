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
from numpy.testing import assert_allclose

from bigdl.keras1.backend import WeightLoader
from bigdl.keras1.converter import DefinitionLoader

np.random.seed(1337)  # for reproducibility
from test.bigdl.test_utils import BigDLTestCase, TestModels


class TestLoadModel(BigDLTestCase):

    def __kmodel_load_def_weight_test(self, kmodel, input_data):
        keras_model_path_json, keras_model_path_hdf5 = self._dump_keras(kmodel, dump_weights=True)
        bmodel = DefinitionLoader.from_json_path(keras_model_path_json)
        WeightLoader.load_weights_from_hdf5(bmodel,
                                            kmodel,
                                            keras_model_path_hdf5)
        bmodel.training(False)
        boutput = bmodel.forward(input_data)
        koutput = kmodel.predict(input_data)
        assert_allclose(boutput, koutput, rtol=1e-5)

    # TODO: These unitest can be unify via reflection.
    def test_load_def_weights_1_layers_graph(self):
        kmodel, input_data, output_data = TestModels.kmodel_graph_1_layer()
        self.__kmodel_load_def_weight_test(kmodel, input_data)

    def test_load_def_weights_4_layers_graph(self):
        kmodel, input_data, output_data = TestModels.kmodel_graph_4_layer()
        self.__kmodel_load_def_weight_test(kmodel, input_data)

    def test_load_def_weights_5_layers_graph(self):
        kmodel, input_data, output_data = TestModels.kmodel_graph_4_layer()
        self.__kmodel_load_def_weight_test(kmodel, input_data)

    # TODO: add load seq model


if __name__ == "__main__":
    pytest.main([__file__])

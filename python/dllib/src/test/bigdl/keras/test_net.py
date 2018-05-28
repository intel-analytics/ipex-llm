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

import pytest

import keras.layers as KLayer
from keras.models import Sequential as KSequential
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import zoo.pipeline.api.keras.layers as ZLayer
from zoo.pipeline.api.keras.models import Model as ZModel
from zoo.pipeline.api.net import Net
from bigdl.nn.layer import Linear, Sigmoid, SoftMax, Model as BModel
from bigdl.util.common import *
from bigdl.nn.layer import Sequential

np.random.seed(1337)  # for reproducibility


class TestLayer(ZooTestCase):

    def test_load_bigdl_model(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        model_path = os.path.join(resource_path, "models/bigdl/bigdl_lenet.model")
        model = Net.load_bigdl(model_path)
        model2 = model.new_graph(["reshape2"])
        model2.freeze_up_to(["pool3"])
        model2.unfreeze()
        import numpy as np
        data = np.zeros([1, 1, 28, 28])
        output = model2.forward(data)
        assert output.shape == (1, 192)

    def test_load_caffe_model(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        model_path = os.path.join(resource_path, "models/caffe/test_persist.caffemodel")
        def_path = os.path.join(resource_path, "models/caffe/test_persist.prototxt")
        model = Net.load_caffe(def_path, model_path)
        model2 = model.new_graph(["ip"])
        model2.freeze_up_to(["conv2"])
        model2.unfreeze()

    def test_load(self):
        input = ZLayer.Input(shape=(5,))
        output = ZLayer.Dense(10)(input)
        zmodel = ZModel(input, output, name="graph1")

        tmp_path = create_tmp_path()
        zmodel.saveModel(tmp_path, None, True)

        model_reloaded = Net.load(tmp_path)

        input_data = np.random.random([3, 5])
        self.compare_output_and_grad_input(zmodel, model_reloaded, input_data)

    def test_load_keras(self):
        model = KSequential()
        model.add(KLayer.Dense(32, activation='relu', input_dim=100))

        tmp_path_json = create_tmp_path() + ".json"
        model_json = model.to_json()
        with open(tmp_path_json, "w") as json_file:
            json_file.write(model_json)
        reloaded_json_model = Net.load_keras(json_path=tmp_path_json)

        tmp_path_hdf5 = create_tmp_path() + ".h5"
        model.save(tmp_path_hdf5)
        reloaded_hdf5_model = Net.load_keras(hdf5_path=tmp_path_hdf5)

    def test_load_tf(self):
        linear = Linear(10, 2)()
        sigmoid = Sigmoid()(linear)
        softmax = SoftMax().set_name("output")(sigmoid)
        model = BModel(linear, softmax)
        input = np.random.random((4, 10))

        tmp_path = create_tmp_path() + "/model.pb"

        model.save_tensorflow([("input", [4, 10])], tmp_path)

        model_reloaded = Net.load_tf(tmp_path, ["input"], ["output"])
        expected_output = model.forward(input)
        output = model_reloaded.forward(input)
        self.assert_allclose(output, expected_output)

    def test_layers_method(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        model_path = os.path.join(resource_path, "models/bigdl/bigdl_lenet.model")
        model = Net.load_bigdl(model_path)
        assert len(model.layers) == 12

    def test_flatten_layers_method(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        model_path = os.path.join(resource_path, "models/bigdl/bigdl_lenet.model")
        model = Net.load_bigdl(model_path)

        assert len(Sequential().add(model).flattened_layers()) == 12


if __name__ == "__main__":
    pytest.main([__file__])

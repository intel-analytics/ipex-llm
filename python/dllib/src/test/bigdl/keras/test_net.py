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
from zoo.pipeline.api.keras.models import Sequential as ZSequential
from zoo.pipeline.api.net import Net, TFNet
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

    def test_deprecated_save(self):
        with pytest.raises(Exception) as e_info:
            input = ZLayer.Input(shape=(5,))
            output = ZLayer.Dense(10)(input)
            zmodel = ZModel(input, output, name="graph1")
            zmodel.save(create_tmp_path())

    def test_save_load_Model(self):
        input = ZLayer.Input(shape=(5,))
        output = ZLayer.Dense(10)(input)
        zmodel = ZModel(input, output, name="graph1")
        tmp_path = create_tmp_path()
        zmodel.saveModel(tmp_path, None, True)
        model_reloaded = Net.load(tmp_path)
        input_data = np.random.random([10, 5])
        y = np.random.random([10, 10])
        model_reloaded.compile(optimizer="adam",
                               loss="mse")
        model_reloaded.fit(x=input_data, y=y, batch_size=8, nb_epoch=2)

    def test_save_load_Sequential(self):
        zmodel = ZSequential()
        dense = ZLayer.Dense(10, input_dim=5)
        zmodel.add(dense)
        tmp_path = create_tmp_path()
        zmodel.saveModel(tmp_path, None, True)
        model_reloaded = Net.load(tmp_path)
        input_data = np.random.random([10, 5])
        y = np.random.random([10, 10])
        model_reloaded.compile(optimizer="adam",
                               loss="mse")
        model_reloaded.fit(x=input_data, y=y, batch_size=8, nb_epoch=1)

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
        zmodel = Net.load_keras(json_path=tmp_path_json)
        assert isinstance(zmodel, Sequential)

        tmp_path_hdf5 = create_tmp_path() + ".h5"
        model.save(tmp_path_hdf5)
        zmodel2 = Net.load_keras(hdf5_path=tmp_path_hdf5)
        assert isinstance(zmodel2, Sequential)

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

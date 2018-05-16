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

import numpy as np

from test.zoo.pipeline.utils.test_utils import ZooTestCase
import os
from zoo.pipeline.api.net import Net

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

if __name__ == "__main__":
    pytest.main([__file__])

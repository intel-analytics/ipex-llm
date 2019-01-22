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

import os
import pytest
import numpy as np

from bigdl.dataset.base import maybe_download
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.inference import InferenceModel

np.random.seed(1337)  # for reproducibility

resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
data_url = "https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/openvino"


class TestInferenceModel(ZooTestCase):
    def test_load_model(self):
        model = InferenceModel(3)
        model.load(os.path.join(resource_path, "models/bigdl/bigdl_lenet.model"))
        input_data = np.random.random([4, 28, 28, 1])
        output_data = model.predict(input_data)

    def test_load_caffe(self):
        model = InferenceModel(10)
        model.load_caffe(os.path.join(resource_path, "models/caffe/test_persist.prototxt"),
                         os.path.join(resource_path, "models/caffe/test_persist.caffemodel"))
        input_data = np.random.random([4, 3, 8, 8])
        output_data = model.predict(input_data)

    def test_load_openvino(self):
        local_path = self.create_temp_dir()
        url = data_url + "/IR_faster_rcnn_resnet101_coco_2018_01_28"
        maybe_download("frozen_inference_graph.xml",
                       local_path, url + "/frozen_inference_graph.xml")
        maybe_download("frozen_inference_graph.bin",
                       local_path, url + "/frozen_inference_graph.bin")
        model = InferenceModel()
        model.load_openvino(local_path + "/frozen_inference_graph.xml",
                            local_path + "/frozen_inference_graph.bin")
        input_data = np.random.random([1, 1, 3, 600, 600])
        output_data = model.predict(input_data)

    def test_load_tf_openvino(self):
        local_path = self.create_temp_dir()
        url = data_url + "/TF_faster_rcnn_resnet101_coco_2018_01_28"
        maybe_download("frozen_inference_graph.pb", local_path, url + "/frozen_inference_graph.pb")
        maybe_download("pipeline.config", local_path, url + "/pipeline.config")
        maybe_download("faster_rcnn_support.json", local_path, url + "/faster_rcnn_support.json")
        model = InferenceModel(3)
        model.load_tf(local_path + "/frozen_inference_graph.pb", backend="openvino",
                      ov_pipeline_config_path=local_path + "/pipeline.config",
                      ov_extensions_config_path=local_path + "/faster_rcnn_support.json")
        input_data = np.random.random([4, 1, 3, 600, 600])
        output_data = model.predict(input_data)
        model2 = InferenceModel(5)
        model2.load_tf(local_path + "/frozen_inference_graph.pb", backend="openvino",
                       model_type="faster_rcnn_resnet101_coco")
        output_data2 = model2.predict(input_data)


if __name__ == "__main__":
    pytest.main([__file__])

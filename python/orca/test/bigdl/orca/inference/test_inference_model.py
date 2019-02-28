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

import tarfile

np.random.seed(1337)  # for reproducibility

resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
data_url = "http://download.tensorflow.org"


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

    def test_load_tf_openvino(self):
        local_path = self.create_temp_dir()
        url = data_url + "/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz"
        file_abs_path = maybe_download("faster_rcnn_resnet101_coco_2018_01_28.tar.gz",
                                       local_path, url)
        tar = tarfile.open(file_abs_path, "r:gz")
        extracted_to = os.path.join(local_path, "faster_rcnn_resnet101_coco_2018_01_28")
        if not os.path.exists(extracted_to):
            print("Extracting %s to %s" % (file_abs_path, extracted_to))
            tar.extractall(local_path)
            tar.close()
        model = InferenceModel(3)
        model.load_tf(model_path=extracted_to + "/frozen_inference_graph.pb",
                      backend="openvino",
                      model_type="faster_rcnn_resnet101_coco",
                      ov_pipeline_config_path=extracted_to + "/pipeline.config",
                      ov_extensions_config_path=None)
        input_data = np.random.random([4, 1, 3, 600, 600])
        output_data = model.predict(input_data)


if __name__ == "__main__":
    pytest.main([__file__])

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
property_path = os.path.join(os.path.split(__file__)[0],
                             "../../../../../zoo/target/classes/app.properties")
data_url = "http://download.tensorflow.org"
s3_url = "https://s3-ap-southeast-1.amazonaws.com"
with open(property_path) as f:
    for _ in range(2):  # skip the first two lines
        next(f)
    for line in f:
        if "data-store-url" in line:
            line = line.strip()
            data_url = line.split("=")[1].replace("\\", "")


class TestInferenceModel(ZooTestCase):

    def test_load_bigdl(self):
        model = InferenceModel(3)
        model.load_bigdl(os.path.join(resource_path, "models/bigdl/bigdl_lenet.model"))
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
        model = InferenceModel(1)
        model_url = s3_url + "/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.xml"
        weight_url = s3_url + "/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.bin"
        model_path = maybe_download("resnet_v1_50.xml",
                                    local_path, model_url)
        weight_path = maybe_download("resnet_v1_50.bin",
                                     local_path, weight_url)
        model.load_openvino(model_path, weight_path)
        input_data = np.random.random([4, 1, 224, 224, 3])
        model.predict(input_data)

    def test_load_tf_openvino_od(self):
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
        model2 = InferenceModel(3)
        model2.load_tf_object_detection_as_openvino(
            model_path=extracted_to + "/frozen_inference_graph.pb",
            object_detection_model_type="faster_rcnn_resnet101_coco",
            pipeline_config_path=extracted_to + "/pipeline.config",
            extensions_config_path=None)
        model2.predict(input_data)

    # def test_load_tf_openvino_ic(self):
    #     local_path = self.create_temp_dir()
    #     print(local_path)
    #     url = data_url + "/models/resnet_v1_50_2016_08_28.tar.gz"
    #     file_abs_path = maybe_download("resnet_v1_50_2016_08_28.tar.gz", local_path, url)
    #     tar = tarfile.open(file_abs_path, "r:gz")
    #     print("Extracting %s to %s" % (file_abs_path, local_path))
    #     tar.extractall(local_path)
    #     tar.close()
    #     model = InferenceModel(3)
    #     model.load_tf_image_classification_as_openvino(
    #         model_path=None,
    #         image_classification_model_type="resnet_v1_50",
    #         checkpoint_path=local_path + "/resnet_v1_50.ckpt",
    #         input_shape=[4, 224, 224, 3],
    #         if_reverse_input_channels=True,
    #         mean_values=[123.68, 116.78, 103.94],
    #         scale=1)
    #     print(model)
    #     input_data = np.random.random([4, 1, 224, 224, 3])
    #     model.predict(input_data)


if __name__ == "__main__":
    pytest.main([__file__])

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
import os
import tarfile
import tempfile
import shutil
from unittest import TestCase

import numpy as np
from bigdl.dllib.feature.dataset.base import maybe_download

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.data import SparkXShards
from bigdl.orca.learn.openvino import Estimator

# property_path = os.path.join(os.path.split(__file__)[0],
#                              "../../../../../../zoo/target/classes/app.properties")
data_url = "https://sourceforge.net/projects/analytics-zoo/files/"
resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
local_path = "/tmp/tmpyhny717gopenvino"

# with open(property_path) as f:
#     for _ in range(2):  # skip the first two lines
#         next(f)
#     for line in f:
#         if "inner-ftp-uri" in line:
#             line = line.strip()
#             data_url = line.split("=")[1].replace("\\", "")


def read_file_and_cast(file_path):
    with open(file_path, "r") as file:
        d = file.readline()
        data_list = list(map(lambda s: float(s), d.split(", ")))
        return data_list


class TestEstimatorForOpenVINO(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        shutil.rmtree(local_path)

    def check_result(self, result, length=0):
        if length == 0:
            length = len(result)
        return np.all(list(map(lambda i: np.allclose(result[i], self.output), range(0, length))))

    def load_resnet(self):
        input_file_path = os.path.join(resource_path, "orca/learn/resnet_input")
        output_file_path = os.path.join(resource_path, "orca/learn/resnet_output")
        self.input = read_file_and_cast(input_file_path)
        self.output = read_file_and_cast(output_file_path)
        self.input = np.array(self.input).reshape([3, 224, 224])
        self.output = np.array(self.output).reshape([4, 1000])[:1]

        os.makedirs(local_path, exist_ok=True)
        model_url = data_url + "/analytics-zoo-data/openvino2020_resnet50.tar"
        model_path = maybe_download("openvino2020_resnet50.tar",
                                    local_path, model_url)
        with tarfile.open(model_path) as tar:
            tar.extractall(path=local_path)
        model_path = os.path.join(local_path, "openvino2020_resnet50/resnet_v1_50.xml")
        self.est = Estimator.from_openvino(model_path=model_path)

    def load_roberta(self):
        os.makedirs(local_path, exist_ok=True)
        model_url = data_url + "/analytics-zoo-data/roberta.tar"
        model_path = maybe_download("roberta.tar",
                                    local_path, model_url)
        with tarfile.open(model_path) as tar:
            tar.extractall(path=local_path)
        model_path = os.path.join(local_path, "roberta/model.xml")
        self.est = Estimator.from_openvino(model_path=model_path)

    def load_multi_output_model(self):
        os.makedirs(local_path, exist_ok=True)
        model_url = data_url + "/analytics-zoo-data/ov_multi_output.tar"
        model_path = maybe_download("ov_multi_output.tar",
                                    local_path, model_url)
        with tarfile.open(model_path) as tar:
            tar.extractall(path=local_path)
        model_path = os.path.join(local_path, "FP32/model_float32.xml")
        self.est = Estimator.from_openvino(model_path=model_path)

    def test_openvino_predict_ndarray(self):
        self.load_resnet()
        input_data = np.array([self.input] * 22)
        input_data = np.concatenate([input_data, np.zeros([1, 3, 224, 224])])
        result = self.est.predict(input_data, batch_size=4)
        assert isinstance(result, np.ndarray)
        assert result.shape == (23, 1000)
        assert self.check_result(result, 22)
        assert not self.check_result(result[22:], 1)

    def test_openvino_predict_ndarray_multi_input(self):
        self.load_roberta()
        input_data = [np.zeros([32, 128]), np.ones([32, 128]), np.zeros([32, 128])]
        result = self.est.predict(input_data, batch_size=5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 2)
        with self.assertRaises(Exception):
            self.est.predict(input_data, feature_cols=["feature"],
                             input_cols=["a", 'input_ids', 'token_type_ids'])
        result2 = self.est.predict(input_data,
                                   input_cols=['input_ids', 'token_type_ids', 'attention_mask'])
        assert isinstance(result2, np.ndarray)
        assert result2.shape == (32, 2)
        assert not np.allclose(result, result2)

    def test_openvino_predict_xshards(self):
        self.load_resnet()
        input_data_list = [np.array([self.input] * 4),
                           np.concatenate([np.array([self.input] * 2),
                                           np.zeros([1, 3, 224, 224])])]
        sc = init_nncontext()
        rdd = sc.parallelize(input_data_list, numSlices=2)
        shards = SparkXShards(rdd)

        def pre_processing(images):
            return {"x": images}

        shards = shards.transform_shard(pre_processing)
        result = self.est.predict(shards)
        result_c = result.collect()
        assert isinstance(result, SparkXShards)
        assert result_c[0].shape == (4, 1000)
        assert result_c[1].shape == (3, 1000)
        assert self.check_result(result_c[0], 4)
        assert self.check_result(result_c[1], 2)
        assert not self.check_result(result_c[1][2:], 1)

    def test_openvino_predict_spark_df(self):
        from pyspark.sql import SparkSession

        self.load_resnet()
        sc = init_nncontext()
        spark = SparkSession(sc)
        input_list = self.input.tolist()
        rdd = sc.range(0, 18, numSlices=5)
        input_df = rdd.map(lambda x: [input_list]).toDF(["feature"])
        result_df = self.est.predict(input_df, feature_cols=["feature"])
        result = list(map(lambda row: np.array(row["resnet_v1_50/predictions/Softmax"]),
                          result_df.select("resnet_v1_50/predictions/Softmax").collect()))
        assert np.array(result_df.select("resnet_v1_50/predictions/Softmax").first()).shape \
               == (1, 1, 1000)
        assert result_df.count() == 18
        assert self.check_result(result, 18)

    def test_openvino_multi_output(self):
        from pyspark.sql import SparkSession
        from bigdl.orca.learn.utils import dataframe_to_xshards

        self.load_multi_output_model()
        sc = init_nncontext()
        spark = SparkSession(sc)
        data = np.random.rand(3, 550, 550)
        rdd = sc.range(0, 2, numSlices=1)
        df = rdd.map(lambda x: [data.tolist()]).toDF(["input"])
        result_df = self.est.predict(df, feature_cols=["input"])
        df_c = result_df.rdd.map(lambda row: [row[1], row[2], row[3], row[4]]).collect()
        df_c = [np.concatenate((np.array(df_c[0][i]), np.array(df_c[1][i]))) for i in range(4)]
        shards, _ = dataframe_to_xshards(df,
                                         validation_data=None,
                                         feature_cols=["input"],
                                         label_cols=None,
                                         mode="predict")
        result_shard = self.est.predict(shards, batch_size=4)
        shard_c = result_shard.collect()[0]
        nd_input = np.squeeze(np.array(df.select('input').collect()))
        result_np = self.est.predict(nd_input)
        assert np.all([np.allclose(r1, r2) for r1, r2 in zip(df_c, result_np)])
        assert np.all([np.allclose(r1, r2) for r1, r2 in zip(shard_c, result_np)])


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])

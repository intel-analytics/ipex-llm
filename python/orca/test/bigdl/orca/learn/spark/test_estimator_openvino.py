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
import tempfile
import subprocess
import tarfile
from unittest import TestCase

import numpy as np
from zoo import init_nncontext
from zoo.orca.data import SparkXShards
from zoo.orca.learn.openvino import Estimator
from bigdl.dataset.base import maybe_download

property_path = os.path.join(os.path.split(__file__)[0],
                             "../../../../../../zoo/target/classes/app.properties")
data_url = "http://10.239.45.10:8081/repository/raw"

with open(property_path) as f:
    for _ in range(2):  # skip the first two lines
        next(f)
    for line in f:
        if "inner-ftp-uri" in line:
            line = line.strip()
            data_url = line.split("=")[1].replace("\\", "")


class TestEstimatorForOpenVINO(TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as local_path:
            model_url = data_url + "/analytics-zoo-data/openvino2020_resnet50.tar"
            model_path = maybe_download("openvino2020_resnet50.tar",
                                        local_path, model_url)
            tar = tarfile.open(model_path)
            tar.extractall(path=local_path)
            tar.close()
            model_path = os.path.join(local_path, "openvino2020_resnet50/resnet_v1_50.xml")
            self.est = Estimator.from_openvino(model_path=model_path)

    def test_openvino_predict_ndarray(self):
        input_data = np.random.random([20, 4, 3, 224, 224])
        result = self.est.predict(input_data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 4, 1000)

    def test_openvino_predict_xshards(self):
        input_data_list = [np.random.random([1, 4, 3, 224, 224]),
                           np.random.random([2, 4, 3, 224, 224])]
        sc = init_nncontext()
        rdd = sc.parallelize(input_data_list, numSlices=2)
        shards = SparkXShards(rdd)

        def pre_processing(images):
            return {"x": images}

        shards = shards.transform_shard(pre_processing)
        result = self.est.predict(shards)
        result_c = result.collect()
        assert isinstance(result, SparkXShards)
        assert result_c[0]["prediction"].shape == (1, 4, 1000)
        assert result_c[1]["prediction"].shape == (2, 4, 1000)

    def test_openvino_predict_spark_df(self):
        from pyspark.sql import SparkSession

        sc = init_nncontext()
        spark = SparkSession(sc)
        rdd = sc.range(0, 20, numSlices=2)
        input_df = rdd.map(lambda x: (np.random.random([1, 4, 3, 224, 224]).tolist())
                           ).toDF(["feature"])
        result_df = self.est.predict(input_df, feature_cols=["feature"])
        assert np.array(result_df.select("prediction").first()).shape == (1, 4, 1000)
        assert result_df.count() == 20


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])

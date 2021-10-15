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


from unittest import TestCase

import os
import pytest
import numpy as np
from pyspark.sql.types import ArrayType, DoubleType

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data import SparkXShards
from bigdl.orca.data.image.utils import chunks
from bigdl.orca.learn.utils import convert_predict_rdd_to_dataframe, _dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, convert_predict_rdd_to_xshard, update_predict_xshards

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")


class TestUtil(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_orca_context(cores=4)

        def to_array_(v):
            return v.toArray().tolist()

        def flatten_(v):
            result = []
            for elem in v:
                result.extend(elem.toArray().tolist())
            return result

        from pyspark.sql import SparkSession
        spark = SparkSession(self.sc)
        spark.udf.register("to_array", to_array_, ArrayType(DoubleType()))
        spark.udf.register("flatten", flatten_, ArrayType(DoubleType()))

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        stop_orca_context()

    def test_convert_predict_rdd_to_dataframe(self):
        rdd = self.sc.range(0, 100)
        df = rdd.map(lambda x: ([float(x)] * 50,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])
        pred_rdd = rdd.map(lambda x: np.array([float(x)] * 50))
        result_df = convert_predict_rdd_to_dataframe(df, pred_rdd)
        expr = "sum(cast(feature <> to_array(prediction) as int)) as error"
        assert result_df.selectExpr(expr).first()["error"] == 0

    def test_convert_predict_rdd_to_dataframe_multi_output(self):
        rdd = self.sc.range(0, 100)
        df = rdd.map(lambda x: ([float(x)] * 50,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])
        pred_rdd = rdd.map(lambda x: [np.array([float(x)] * 25), np.array([float(x)] * 25)])
        result_df = convert_predict_rdd_to_dataframe(df, pred_rdd)
        expr = "sum(cast(feature <> flatten(prediction) as int)) as error"
        assert result_df.selectExpr(expr).first()["error"] == 0

    def test_convert_predict_rdd_to_xshard(self):
        rdd = self.sc.range(0, 110).map(lambda x: np.array([x]*50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(lambda x: {"x": np.stack(x)})
        shards = SparkXShards(shards)
        pred_rdd = self.sc.range(0, 110).map(lambda x: np.array([x]*50))
        result_shards = convert_predict_rdd_to_xshard(shards, pred_rdd)
        result = np.concatenate([shard["prediction"] for shard in result_shards.collect()])
        expected_result = np.concatenate([shard["x"] for shard in result_shards.collect()])

        assert np.array_equal(result, expected_result)

    def test_convert_predict_rdd_to_xshard_multi_output(self):
        rdd = self.sc.range(0, 110).map(lambda x: np.array([x]*50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(lambda x: {"x": np.stack(x)})
        shards = SparkXShards(shards)
        pred_rdd = self.sc.range(0, 110).map(lambda x: [np.array([x]*24), np.array([x]*26)])
        result_shards = convert_predict_rdd_to_xshard(shards, pred_rdd)
        result = np.concatenate([np.concatenate(shard["prediction"], axis=1)
                                 for shard in result_shards.collect()])
        expected_result = np.concatenate([shard["x"] for shard in result_shards.collect()])

        assert np.array_equal(result, expected_result)

    def test_update_predict_xshard(self):

        def get_xshards(key):
            rdd = self.sc.range(0, 110).map(lambda x: np.array([x] * 50))
            shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(
                lambda x: {key: np.stack(x)})
            shards = SparkXShards(shards)
            return shards

        data_shards = get_xshards("x")
        pred_shards = get_xshards("prediction")
        result_shards = update_predict_xshards(data_shards, pred_shards)
        result = np.concatenate([shard["prediction"] for shard in result_shards.collect()])
        expected_result = np.concatenate([shard["x"] for shard in result_shards.collect()])

        assert np.array_equal(result, expected_result)

    def test_update_predict_xshard_multi_output(self):

        def get_data_xshards(key):
            rdd = self.sc.range(0, 110).map(lambda x: np.array([x] * 50))
            shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(
                lambda x: {key: np.stack(x)})
            shards = SparkXShards(shards)
            return shards

        def get_pred_xshards(key):
            rdd = self.sc.range(0, 110).map(lambda x: np.array([x] * 50))
            shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(
                lambda x: {key: np.stack(x)}).map(lambda x: {key: [x[key][:, :24], x[key][:, 24:]]})
            shards = SparkXShards(shards)
            return shards

        data_shards = get_data_xshards("x")
        pred_shards = get_pred_xshards("prediction")

        result_shards = update_predict_xshards(data_shards, pred_shards)
        result = np.concatenate([np.concatenate(shard["prediction"], axis=1)
                                 for shard in result_shards.collect()])
        expected_result = np.concatenate([shard["x"] for shard in result_shards.collect()])

        assert np.array_equal(result, expected_result)

    def test_convert_predict_xshards_to_dataframe(self):
        rdd = self.sc.range(0, 100)
        df = rdd.map(lambda x: ([float(x)] * 50,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])

        pred_shards = _dataframe_to_xshards(df, feature_cols=["feature"]).transform_shard(
            lambda x: {"prediction": x["x"]})

        result_df = convert_predict_xshards_to_dataframe(df, pred_shards)
        expr = "sum(cast(feature <> to_array(prediction) as int)) as error"
        assert result_df.selectExpr(expr).first()["error"] == 0

    def test_convert_predict_xshards_to_dataframe_multi_output(self):
        rdd = self.sc.range(0, 100)
        df = rdd.map(lambda x: ([float(x)] * 50,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])
        pred_shards = _dataframe_to_xshards(df, feature_cols=["feature"]).transform_shard(
            lambda x: {"prediction": [x["x"][:, :25], x["x"][:, 25:]]})
        result_df = convert_predict_xshards_to_dataframe(df, pred_shards)
        expr = "sum(cast(feature <> flatten(prediction) as int)) as error"
        assert result_df.selectExpr(expr).first()["error"] == 0

    def test_array2dict(self):
        from bigdl.orca.learn.utils import arrays2dict
        record_num = 100
        shard_size = 30
        data = [(np.float32(np.random.randn(1, 50)), np.float32([np.random.randint(0, 2,)]))
                for i in range(record_num)]
        result = arrays2dict(data, feature_cols=["feature"], label_cols=["label"],
                             shard_size=shard_size)
        for i, d in enumerate(result):
            if (record_num % shard_size == 0) or (i != record_num // shard_size):
                assert d['x'].shape[0] == shard_size
                assert d['y'].shape[0] == shard_size
            else:
                assert d['x'].shape[0] == record_num % shard_size
                assert d['y'].shape[0] == record_num % shard_size

    def test_array2dict_shard_size_none(self):
        from bigdl.orca.learn.utils import arrays2dict
        record_num = 100
        data = [(np.float32(np.random.randn(1, 50)), np.float32([np.random.randint(0, 2,)]))
                for i in range(record_num)]
        result = arrays2dict(data, feature_cols=["feature"], label_cols=["label"], shard_size=None)
        for i, d in enumerate(result):
            assert d['x'].shape[0] == record_num
            assert d['y'].shape[0] == record_num

    def test_dataframe_to_xshards(self):
        rdd = self.sc.range(0, 100)
        df = rdd.map(lambda x: ([float(x)] * 50,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])
        num_partitions = df.rdd.getNumPartitions()
        # test shard_size = None
        shards = _dataframe_to_xshards(df, feature_cols=["feature"], label_cols=["label"])
        num_shards = shards.rdd.count()
        assert num_shards == num_partitions

        from bigdl.orca import OrcaContext
        OrcaContext._shard_size = 1
        shards = _dataframe_to_xshards(df, feature_cols=["feature"], label_cols=["label"])
        num_shards = shards.rdd.count()
        assert num_shards == df.rdd.count()


if __name__ == "__main__":
    pytest.main([__file__])

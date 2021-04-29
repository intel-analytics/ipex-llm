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

import os.path
import pytest
import tempfile
from unittest import TestCase

from pyspark.sql.functions import col, max, min, array
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

from zoo.orca import OrcaContext
from zoo.friesian.feature import FeatureTable, StringIndex
from zoo.common.nncontext import *


class TestTable(TestCase):
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

    def test_fillna_int(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filled_tbl = feature_tbl.fillna(5, ["col_2", "col_3"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_2 is null").count() != 0 and feature_tbl \
            .df.filter("col_3 is null").count() != 0, "feature_tbl should not be changed"
        assert filled_tbl.df.filter("col_2 == 5").count() == 1, "col_2 null values should be " \
                                                                "filled with 5"
        assert filled_tbl.df.filter("col_3 == 5").count() == 1, "col_3 null values should be " \
                                                                "filled with 5"
        filled_tbl = feature_tbl.fillna(5, None)
        assert filled_tbl.df.filter("col_2 == 5").count() == 1, "col_2 null values should be " \
                                                                "filled with 5"
        assert filled_tbl.df.filter("col_3 == 5").count() == 1, "col_3 null values should be " \
                                                                "filled with 5"
        with self.assertRaises(Exception) as context:
            feature_tbl.fillna(0, ["col_2", "col_3", "col_8"])
        self.assertTrue('do not exist in this Table' in str(context.exception))

    def test_fillna_double(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filled_tbl = feature_tbl.fillna(3.2, ["col_2", "col_3"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_2 is null").count() != 0 and feature_tbl \
            .df.filter("col_3 is null").count() != 0, "feature_tbl should not be changed"
        assert filled_tbl.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_3 is null").count() == 0, "col_3 null values should be " \
                                                                   "filled"
        filled_tbl = feature_tbl.fillna(5, ["col_2", "col_3"])
        assert filled_tbl.df.filter("col_2 == 5").count() == 1, "col_2 null values should be " \
                                                                "filled with 5"
        assert filled_tbl.df.filter("col_3 == 5").count() == 1, "col_3 null values should be " \
                                                                "filled with 5"

    def test_fillna_long(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filled_tbl = feature_tbl.fillna(3, ["col_1", "col_2", "col_3"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_2 is null").count() != 0 and feature_tbl \
            .df.filter("col_3 is null").count() != 0, "feature_tbl should not be changed"
        assert filled_tbl.df.filter("col_1 is null").count() == 0, "col_1 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_3 is null").count() == 0, "col_3 null values should be " \
                                                                   "filled"

    def test_fillna_string(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        with self.assertRaises(Exception) as context:
            feature_tbl.fillna(3.2, ["col_4", "col_5"])
        self.assertTrue('numeric does not match the type of column col_4' in str(context.exception))

        filled_tbl = feature_tbl.fillna("bb", ["col_4", "col_5"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert filled_tbl.df.filter("col_4 is null").count() == 0, "col_4 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_5 is null").count() == 0, "col_5 null values should be " \
                                                                   "filled"

    def test_gen_string_idx(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit=1)
        assert string_idx_list[0].size() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].size() == 2, "col_5 should have 2 indices"
        with tempfile.TemporaryDirectory() as local_path:
            for str_idx in string_idx_list:
                str_idx.write_parquet(local_path)
                str_idx_log = str_idx.log(["id"])
                assert str_idx.df.filter("id == 1").count() == 1, "id in str_idx should = 1"
                assert str_idx_log.df.filter("id == 1").count() == 0, "id in str_idx_log should " \
                                                                      "!= 1"
            assert os.path.isdir(local_path + "/col_4.parquet")
            assert os.path.isdir(local_path + "/col_5.parquet")
            new_col_4_idx = StringIndex.read_parquet(local_path + "/col_4.parquet")
            assert "col_4" in new_col_4_idx.df.columns, "col_4 should be a column of new_col_4_idx"
            with self.assertRaises(Exception) as context:
                StringIndex.read_parquet(local_path + "/col_5.parquet", "col_4")
            self.assertTrue('col_4 should be a column of the DataFrame' in str(context.exception))

    def test_gen_string_idx_dict(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit={"col_4": 1,
                                                                                     "col_5": 3})
        with self.assertRaises(Exception) as context:
            feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit="col_4:1,col_5:3")
        self.assertTrue('freq_limit only supports int, dict or None, but get str' in str(
            context.exception))
        assert string_idx_list[0].size() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].size() == 1, "col_5 should have 1 indices"

    def test_gen_string_idx_none(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit=None)
        assert string_idx_list[0].size() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].size() == 2, "col_5 should have 2 indices"

    def test_clip(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        clip_tbl = feature_tbl.clip(["col_1", "col_2", "col_3"], min=2, max=None)
        assert isinstance(clip_tbl, FeatureTable), "clip_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 < 2").count() != 0 and feature_tbl \
            .df.filter("col_2 < 2").count() != 0, "feature_tbl should not be changed"
        assert clip_tbl.df.filter("col_1 < 2").count() == 0, "col_1 should >= 2"
        assert clip_tbl.df.filter("col_2 < 2").count() == 0, "col_2 should >= 2"
        assert clip_tbl.df.filter("col_3 < 2").count() == 0, "col_3 should >= 2"
        with self.assertRaises(Exception) as context:
            feature_tbl.clip(None, 2)
        self.assertTrue('columns should be str or list of str, but got None.'
                        in str(context.exception))

        feature_tbl = FeatureTable.read_parquet(file_path)
        clip_tbl = feature_tbl.clip(["col_1", "col_2", "col_3"], min=None, max=1)
        assert isinstance(clip_tbl, FeatureTable), "clip_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 > 1").count() != 0 and feature_tbl \
            .df.filter("col_2 > 1").count() != 0, "feature_tbl should not be changed"
        assert clip_tbl.df.filter("col_1 > 1").count() == 0, "col_1 should <= 1"
        assert clip_tbl.df.filter("col_2 > 1").count() == 0, "col_2 should <= 1"
        assert clip_tbl.df.filter("col_3 > 1").count() == 0, "col_3 should <= 1"

        feature_tbl = FeatureTable.read_parquet(file_path)
        clip_tbl = feature_tbl.clip(["col_1", "col_2", "col_3"], min=0, max=1)
        assert isinstance(clip_tbl, FeatureTable), "clip_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 > 1 or col_1 < 0").count() != 0 and feature_tbl \
            .df.filter("col_2 > 1 or col_2 < 0").count() != 0, "feature_tbl should not be changed"
        assert clip_tbl.df.filter("col_1 < 0").count() == 0, "col_1 should >= 0"
        assert clip_tbl.df.filter("col_2 > 1").count() == 0, "col_2 should <= 1"
        assert clip_tbl.df.filter("col_3 < 0 or col_3 > 1").count() == 0, "col_3 should >=0 " \
                                                                          "and <= 1"

    def test_dropna(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        dropped_tbl = feature_tbl.dropna(["col_1", "col_4"])
        assert isinstance(dropped_tbl, FeatureTable), "dropped_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 is null").count() != 0 and feature_tbl\
            .df.filter("col_4 is null").count() != 0, "feature_tbl should not be changed"
        assert dropped_tbl.df.filter("col_1 is null").count() == 0, "col_1 null values should " \
                                                                    "be dropped"
        assert dropped_tbl.df.filter("col_4 is null").count() == 0, "col_4 null values should " \
                                                                    "be dropped"
        assert 0 < dropped_tbl.df.count() < feature_tbl.df.count(), "the number of rows should " \
                                                                    "be decreased"

        dropped_tbl = feature_tbl.dropna(["col_1", "col_4"], how="all")
        assert dropped_tbl.df.filter("col_1 is null and col_4 is null").count() == 0, \
            "col_1 and col_4 should not both have null values"
        dropped_tbl = feature_tbl.dropna(["col_2", "col_4"], how="all")
        assert dropped_tbl.df.filter("col_2 is null").count() > 0, \
            "col_2 should still have null values after dropna with how=all"

        dropped_tbl = feature_tbl.dropna(["col_2", "col_3", "col_5"], thresh=2)
        assert dropped_tbl.df.filter("col_2 is null").count() > 0, \
            "col_2 should still have null values after dropna with thresh=2"
        assert dropped_tbl.df.filter("col_3 is null and col_5 is null").count() == 0, \
            "col_3 and col_5 should not both have null values"

    def test_fill_median(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        with self.assertRaises(Exception) as context:
            feature_tbl.fill_median(["col_4", "col_5"])
        self.assertTrue('col_4 with data type StringType is not supported' in
                        str(context.exception))

        filled_tbl = feature_tbl.fill_median(["col_1", "col_2"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert filled_tbl.df.filter("col_1 is null").count() == 0, "col_1 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                   "filled"

    def test_rename(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        name_dict = {"col_1": "new_col1", "col_4": "new_col4"}
        rename_tbl = feature_tbl.rename(name_dict)
        cols = rename_tbl.df.columns
        assert isinstance(rename_tbl, FeatureTable), "rename_tbl should be a FeatureTable"
        assert "col_1" in feature_tbl.df.columns, "feature_tbl should not be changed"
        assert "new_col1" in cols, "new_col1 should be a column of the renamed tbl."
        assert "new_col4" in cols, "new_col4 should be a column of the renamed tbl."

    def test_log(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        log_tbl = feature_tbl.log(["col_1", "col_2", "col_3"])
        assert isinstance(log_tbl, FeatureTable), "log_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 == 1").count() != 0 and feature_tbl \
            .df.filter("col_2 == 1").count() != 0, "feature_tbl should not be changed"
        assert log_tbl.df.filter("col_1 == 1").count() == 0, "col_1 should != 1"
        assert log_tbl.df.filter("col_2 == 1").count() == 0, "col_2 should != 1"
        assert log_tbl.df.filter("col_3 == 1").count() == 0, "col_3 should != 1"

    def test_merge(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        merged_tbl = feature_tbl.merge_cols(["col_1", "col_2", "col_3"], "int_cols")
        assert "col_1" not in merged_tbl.df.columns, "col_1 shouldn't be a column of merged_tbl"
        assert "int_cols" in merged_tbl.df.columns, "int_cols should be a column of merged_tbl"
        assert "col_1" in feature_tbl.df.columns, "col_1 should be a column of feature_tbl"

    def test_norm(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path).fillna(0, ["col_2", "col_3"])
        normalized_tbl = feature_tbl.normalize(["col_2"])
        max_value = normalized_tbl.df.select("col_2") \
            .agg(max(col("col_2")).alias("max")) \
            .rdd.map(lambda row: row['max']).collect()[0]
        min_value = normalized_tbl.df.select("col_2") \
            .agg(min(col("col_2")).alias("min")) \
            .rdd.map(lambda row: row['min']).collect()[0]

        assert max_value <= 1, "col_2 shouldn't be more than 1 after normalization"
        assert min_value >= 0, "col_2 shouldn't be less than 0 after normalization"

        tbl2 = FeatureTable(feature_tbl.df.withColumn("col2-col3", array(["col_2", "col_3"])))
        normalized_tbl2 = tbl2.normalize(["col_2", "col2-col3"])
        normalized_tbl2.compute()

    def test_cross(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path).fillna(0, ["col_2", "col_3"])
        crossed_tbl = feature_tbl.cross_columns([["col_2", "col_3"]], [100])
        assert "col_2_col_3" in crossed_tbl.df.columns, "crossed column is not created"
        max_value = crossed_tbl.df.select("col_2_col_3") \
            .agg(max(col("col_2_col_3")).alias("max")) \
            .rdd.map(lambda row: row['max']).collect()[0]
        min_value = crossed_tbl.df.select("col_2_col_3") \
            .agg(min(col("col_2_col_3")).alias("min")) \
            .rdd.map(lambda row: row['min']).collect()[0]

        assert max_value <= 100, "cross value shouldn't be more than 100 after cross"
        assert min_value > 0, "cross value shouldn't be less than 0 after cross"

    def test_add_negative_items(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", 1, "2019-07-01 12:01:19.000"),
                ("jack", 2, "2019-08-01 12:01:19.000"),
                ("jack", 3, "2019-09-01 12:01:19.000"),
                ("alice", 4, "2019-09-01 12:01:19.000"),
                ("alice", 5, "2019-10-01 12:01:19.000"),
                ("alice", 6, "2019-11-01 12:01:19.000")]
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("item", IntegerType(), True),
            StructField("time", StringType(), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)
        tbl = FeatureTable(df).add_negative_samples(10)
        dft = tbl.df
        assert tbl.size() == 12
        assert dft.filter("label == 1").count() == 6
        assert dft.filter("label == 0").count() == 6

    def test_add_hist_seq(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", 1, "2019-07-01 12:01:19.000"),
                ("jack", 2, "2019-08-01 12:01:19.000"),
                ("jack", 3, "2019-09-01 12:01:19.000"),
                ("jack", 4, "2019-07-02 12:01:19.000"),
                ("jack", 5, "2019-08-03 12:01:19.000"),
                ("jack", 6, "2019-07-04 12:01:19.000"),
                ("jack", 7, "2019-08-05 12:01:19.000"),
                ("alice", 4, "2019-09-01 12:01:19.000"),
                ("alice", 5, "2019-10-01 12:01:19.000"),
                ("alice", 6, "2019-11-01 12:01:19.000")]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("item", IntegerType(), True),
                             StructField("time", StringType(), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.withColumn("ts", col("time").cast("timestamp").cast("long"))
        tbl = FeatureTable(df.select("name", "item", "ts")) \
            .add_hist_seq("name", ["item"], "ts", 1, 4)
        assert tbl.size() == 8
        assert tbl.df.filter(col("name") == "alice").count() == 2
        assert tbl.df.filter("name like '%jack'").count() == 6
        assert "item_hist_seq" in tbl.df.columns

    def test_gen_neg_hist_seq(self):
        spark = OrcaContext.get_spark_session()
        sc = OrcaContext.get_spark_context()
        data = [
            ("jack", [1, 2, 3, 4, 5]),
            ("alice", [4, 5, 6, 7, 8]),
            ("rose", [1, 2])]
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("item_hist_seq", ArrayType(IntegerType()), True)])

        df = spark.createDataFrame(data, schema)
        df2 = sc \
            .parallelize([(1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 2), (8, 2), (9, 2)]) \
            .toDF(["item", "category"]).withColumn("item", col("item").cast("Integer")) \
            .withColumn("category", col("category").cast("Integer"))
        tbl = FeatureTable(df)
        tbl = tbl.add_neg_hist_seq(9, "item_hist_seq", 4)
        assert tbl.df.select("neg_item_hist_seq").count() == 3

    def test_gen_cats_from_items(self):
        spark = OrcaContext.get_spark_session()
        sc = OrcaContext.get_spark_context()
        data = [
            ("jack", [1, 2, 3, 4, 5]),
            ("alice", [4, 5, 6, 7, 8]),
            ("rose", [1, 2])]
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("item_hist_seq", ArrayType(IntegerType()), True)])

        df = spark.createDataFrame(data, schema)
        df.filter("name like '%alice%'").show()

        df2 = sc \
            .parallelize([(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (8, 2), (9, 2)]) \
            .toDF(["item", "category"]).withColumn("item", col("item").cast("Integer")) \
            .withColumn("category", col("category").cast("Integer"))
        tbl = FeatureTable(df)
        tbl2 = tbl.add_neg_hist_seq(9, "item_hist_seq", 4)
        tbl3 = tbl2.add_feature(["item_hist_seq", "neg_item_hist_seq"], FeatureTable(df2), 5)
        assert tbl3.df.select("category_hist_seq").count() == 3
        assert tbl3.df.select("neg_category_hist_seq").count() == 3
        assert tbl3.df.filter("name like '%alice%'").select("neg_category_hist_seq").count() == 1
        assert tbl3.df.filter("name == 'rose'").select("neg_category_hist_seq").count() == 1

    def test_pad(self):
        spark = OrcaContext.get_spark_session()
        data = [
            ("jack", [1, 2, 3, 4, 5], [[1, 2, 3], [1, 2, 3]]),
            ("alice", [4, 5, 6, 7, 8], [[1, 2, 3], [1, 2, 3]]),
            ("rose", [1, 2], [[1, 2, 3]])]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("list", ArrayType(IntegerType()), True),
                             StructField("matrix", ArrayType(ArrayType(IntegerType())))])
        df = spark.createDataFrame(data, schema)
        tbl = FeatureTable(df).pad(["list", "matrix"], 4)
        dft = tbl.df
        assert dft.filter("size(matrix) = 4").count() == 3
        assert dft.filter("size(list) = 4").count() == 3

    def test_mask(self):
        spark = OrcaContext.get_spark_session()
        data = [
            ("jack", [1, 2, 3, 4, 5]),
            ("alice", [4, 5, 6, 7, 8]),
            ("rose", [1, 2])]
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("history", ArrayType(IntegerType()), True)])

        df = spark.createDataFrame(data, schema)
        tbl = FeatureTable(df).mask(["history"], 4)
        assert "history_mask" in tbl.df.columns
        assert tbl.df.filter("size(history_mask) = 4").count() == 3
        assert tbl.df.filter("size(history_mask) = 2").count() == 0

    def test_add_length(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", [1, 2, 3, 4, 5]),
                ("alice", [4, 5, 6, 7, 8]),
                ("rose", [1, 2])]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("history", ArrayType(IntegerType()), True)])

        df = spark.createDataFrame(data, schema)
        tbl = FeatureTable(df)
        tbl = tbl.add_length("history")
        assert "history_length" in tbl.df.columns
        assert tbl.df.filter("history_length = 5").count() == 2
        assert tbl.df.filter("history_length = 2").count() == 1

    def test_median(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        with self.assertRaises(Exception) as context:
            feature_tbl.median(["col_4", "col_5"])
        self.assertTrue('col_4 with data type StringType is not supported' in
                        str(context.exception))

        median_tbl = feature_tbl.median(["col_1", "col_2", "col_3"])
        assert isinstance(median_tbl, FeatureTable), "median_tbl should be a FeatureTable"
        assert median_tbl.df.count() == 3, "the number of rows of median_tbl should be equal to " \
                                           "the number of specified columns"
        assert median_tbl.df.filter("column == 'col_1'").count() == 1, "col_1 should exist in " \
                                                                       "'column' of median_tbl"
        assert median_tbl.df.filter("column == 'col_2'").filter("median == 1.0").count() == 1, \
            "the median of col_2 should be 1.0"


if __name__ == "__main__":
    pytest.main([__file__])

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
import shutil

import pytest
import tempfile
from unittest import TestCase

from zoo.common.nncontext import *
from zoo.friesian.feature import FeatureTable, StringIndex


class TestTable(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

    def test_fillna_int(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filled_tbl = feature_tbl.fillna(0, ["col_2", "col_3"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_2 is null").count() != 0 and feature_tbl\
            .df.filter("col_3 is null").count() != 0, "feature_tbl should not be changed"
        assert filled_tbl.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_3 is null").count() == 0, "col_3 null values should be " \
                                                                   "filled"

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
        assert string_idx_list[0].count() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].count() == 2, "col_5 should have 2 indices"
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
        assert string_idx_list[0].count() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].count() == 1, "col_5 should have 1 indices"

    def test_gen_string_idx_none(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit=None)
        assert string_idx_list[0].count() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].count() == 2, "col_5 should have 2 indices"

    def test_clip(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        clip_tbl = feature_tbl.clip(["col_1", "col_2", "col_3"], 2)
        assert isinstance(clip_tbl, FeatureTable), "clip_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 < 2").count() != 0 and feature_tbl \
            .df.filter("col_2 < 2").count() != 0, "feature_tbl should not be changed"
        assert clip_tbl.df.filter("col_1 < 2").count() == 0, "col_1 should >= 2"
        assert clip_tbl.df.filter("col_2 < 2").count() == 0, "col_2 should >= 2"
        assert clip_tbl.df.filter("col_3 < 2").count() == 0, "col_3 should >= 2"

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

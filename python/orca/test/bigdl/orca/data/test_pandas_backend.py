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

import os.path
import shutil

import numpy as np
import pytest
from unittest import TestCase

import bigdl.orca.data
import bigdl.orca.data.pandas
from bigdl.orca import OrcaContext
from bigdl.orca.data import SharedValue
from bigdl.dllib.nncontext import *
from bigdl.orca.data import SparkXShards


class TestXShardsPandasBackend(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        OrcaContext.pandas_read_backend = "pandas"

    def tearDown(self):
        OrcaContext.pandas_read_backend = "spark"

    def test_read_local_csv(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"
        file_path = os.path.join(self.resource_path, "abc")
        with self.assertRaises(Exception) as context:
            xshards = bigdl.orca.data.pandas.read_csv(file_path)
        self.assertTrue('No such file or directory' in str(context.exception))
        file_path = os.path.join(self.resource_path, "image3d")
        with self.assertRaises(Exception) as context:
            xshards = bigdl.orca.data.pandas.read_csv(file_path)
        # This error is raised by pandas.errors.ParserError
        self.assertTrue('Error tokenizing data' in str(context.exception), str(context.exception))

    def test_read_local_json(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = bigdl.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "value" in df.columns, "value is not in columns"

    def test_read_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/nyc_taxi.csv"
            data_shard = bigdl.orca.data.pandas.read_csv(file_path)
            data = data_shard.collect()
            df = data[0]
            assert "value" in df.columns, "value is not in columns"

    def test_read_csv_with_args(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, sep=',', header=0)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"

    def test_save(self):
        temp = tempfile.mkdtemp()
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        path = os.path.join(temp, "data.pkl")
        data_shard.save_pickle(path)
        shards = bigdl.orca.data.XShards.load_pickle(path)
        assert isinstance(shards, bigdl.orca.data.SparkXShards)
        shutil.rmtree(temp)

    def test_get_item(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = bigdl.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        selected_shard = data_shard["value"]
        assert data_shard.is_cached(), "data_shard should be cached"
        assert not selected_shard.is_cached(), "selected_shard should not be cached"
        data1 = data_shard.collect()
        data2 = selected_shard.collect()
        assert data1[0]["value"].values[0] == data2[0][0], "value should be same"
        assert data1[1]["value"].values[0] == data2[1][0], "value should be same"
        with self.assertRaises(Exception) as context:
            len(data_shard['abc'])
        self.assertTrue('Invalid key for this XShards' in str(context.exception))

if __name__ == "__main__":
    pytest.main([__file__])

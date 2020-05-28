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

import zoo.orca.data
import zoo.orca.data.pandas
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.common.nncontext import *


class TestSparkXShards(ZooTestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        sparkConf = init_spark_conf().setMaster("local[4]").setAppName("testSparkXShards")
        self.sc = init_nncontext(sparkConf)

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_read_local_csv(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"

    def test_read_local_json(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, self.sc,
                                                    orient='columns', lines=True)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "value" in df.columns, "value is not in columns"

    def test_read_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/nyc_taxi.csv"
            data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)
            data = data_shard.collect()
            df = data[0]
            assert "value" in df.columns, "value is not in columns"

    def test_repartition(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, self.sc)
        partitions_num_1 = data_shard.rdd.getNumPartitions()
        assert partitions_num_1 == 4, "number of partition should be 4"
        data_shard.repartition(1)
        partitions_num_2 = data_shard.rdd.getNumPartitions()
        assert partitions_num_2 == 1, "number of partition should be 1"

    def test_apply(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, self.sc,
                                                    orient='columns', lines=True)
        data = data_shard.collect()
        assert data[0]["value"].values[0] > 0, "value should be positive"

        def negative(df, column_name):
            df[column_name] = df[column_name] * (-1)
            return df

        data_shard.transform_shard(negative, "value")
        data2 = data_shard.collect()
        assert data2[0]["value"].values[0] < 0, "value should be negative"

    def test_read_csv_with_args(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc, sep=',', header=0)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"

    def test_partition_by_single_column(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)
        data_shard.partition_by(cols="location")
        partitions = data_shard.rdd.glom().collect()
        assert len(partitions) == 4

        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)
        data_shard.partition_by(cols="location", num_partitions=3)
        partitions = data_shard.rdd.glom().collect()
        assert len(partitions) == 3

    def test_unique(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)
        location_list = data_shard.unique("location")
        assert len(location_list) == 6

    def test_split(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)
        data_shard.transform_shard(lambda df: (df[0:-1], df[-1:]))
        shards_splits = data_shard.split()
        assert len(shards_splits) == 2
        data1 = shards_splits[0].collect()
        data2 = shards_splits[1].collect()
        assert len(data1[0].index) > 1
        assert len(data2[0].index) == 1

    def test_save(self):
        temp = tempfile.mkdtemp()
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.sc)
        path = os.path.join(temp, "data.pkl")
        data_shard.save_pickle(path)
        shards = zoo.orca.data.SparkXShards.load_pickle(path, self.sc)
        assert isinstance(shards, zoo.orca.data.SparkXShards)
        shutil.rmtree(temp)


if __name__ == "__main__":
    pytest.main([__file__])

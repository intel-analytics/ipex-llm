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
import ray

import zoo.orca.data.pandas
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from test.zoo.orca.data.conftest import get_ray_ctx


class TestDataShards(ZooTestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        self.ray_ctx = get_ray_ctx()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        pass

    def test_read_local_csv(self):
        file_path = os.path.join(self.resource_path, "orca/data")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.ray_ctx)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"

    def test_read_local_json(self):
        file_path = os.path.join(self.resource_path, "orca/data")
        data_shard = zoo.orca.data.pandas.read_json(file_path, self.ray_ctx, orient='columns',
                                                    lines=True)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "value" in df.columns, "value is not in columns"

    def test_read_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/nyc_taxi.csv"
            data_shard = zoo.orca.data.pandas.read_csv(file_path, self.ray_ctx)
            data = data_shard.collect()
            df = data[0]
            assert "value" in df.columns, "value is not in columns"

    def test_repartition(self):
        file_path = os.path.join(self.resource_path, "orca/data")
        data_shard = zoo.orca.data.pandas.read_json(file_path, self.ray_ctx)
        partitions1 = data_shard.get_partitions()
        assert len(partitions1) == 2, "number of partition should be 2"
        data_shard.repartition(1)
        partitions2 = data_shard.get_partitions()
        assert len(partitions2) == 1, "number of partition should be 1"
        partition_data = ray.get(partitions2[0].get_data())
        assert len(partition_data) == 2, "partition 0 should have 2 objects"

    def test_apply(self):
        file_path = os.path.join(self.resource_path, "orca/data")
        data_shard = zoo.orca.data.pandas.read_json(file_path, self.ray_ctx, orient='columns',
                                                    lines=True)
        data = data_shard.collect()
        assert data[0]["value"].values[0] > 0, "value should be positive"

        def negative(df, column_name):
            df[column_name] = df[column_name] * (-1)
            return df

        data_shard.apply(negative, "value")
        data2 = data_shard.collect()
        assert data2[0]["value"].values[0] < 0, "value should be negative"

    def test_read_csv_with_args(self):
        file_path = os.path.join(self.resource_path, "orca/data")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, self.ray_ctx, sep=',', header=0)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"


if __name__ == "__main__":
    pytest.main([__file__])

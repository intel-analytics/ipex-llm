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
from unittest import TestCase

import zoo.orca.data
import zoo.orca.data.pandas
from zoo.orca.data import SharedValue
from zoo.common.nncontext import *


class TestSparkXShards(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

    def test_read_local_csv(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"
        file_path = os.path.join(self.resource_path, "abc")
        with self.assertRaises(Exception) as context:
            xshards = zoo.orca.data.pandas.read_csv(file_path)
        self.assertTrue('No such file or directory' in str(context.exception))
        file_path = os.path.join(self.resource_path, "image3d")
        with self.assertRaises(Exception) as context:
            xshards = zoo.orca.data.pandas.read_csv(file_path)
        # This error is raised by pandas.errors.ParserError
        self.assertTrue('Error tokenizing data' in str(context.exception))

    def test_read_local_json(self):
        ZooContext.orca_pandas_read_backend = "pandas"
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "value" in df.columns, "value is not in columns"

    def test_read_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/nyc_taxi.csv"
            data_shard = zoo.orca.data.pandas.read_csv(file_path)
            data = data_shard.collect()
            df = data[0]
            assert "value" in df.columns, "value is not in columns"

    def test_repartition(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        partitions_num_1 = data_shard.rdd.getNumPartitions()
        assert partitions_num_1 == 2, "number of partition should be 2"
        data_shard.cache()
        partitioned_shard = data_shard.repartition(1)
        assert data_shard.is_cached(), "data_shard should be cached"
        assert partitioned_shard.is_cached(), "partitioned_shard should be cached"
        data_shard.uncache()
        assert not data_shard.is_cached(), "data_shard should be uncached"
        partitions_num_2 = partitioned_shard.rdd.getNumPartitions()
        assert partitions_num_2 == 1, "number of partition should be 1"

    def test_apply(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data = data_shard.collect()
        assert data[0]["value"].values[0] > 0, "value should be positive"

        def negative(df, column_name):
            df[column_name] = df[column_name] * (-1)
            return df

        trans_data_shard = data_shard.transform_shard(negative, "value")
        data2 = trans_data_shard.collect()
        assert data2[0]["value"].values[0] < 0, "value should be negative"

    def test_read_csv_with_args(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path, sep=',', header=0)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"

    def test_partition_by_single_column(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        partitioned_shard = data_shard.partition_by(cols="location", num_partitions=4)
        partitions = partitioned_shard.rdd.glom().collect()
        assert len(partitions) == 4

        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        partitioned_shard = data_shard.partition_by(cols="location", num_partitions=3)
        assert not data_shard.is_cached(), "data_shard should be uncached"
        assert partitioned_shard.is_cached(), "partitioned_shard should be cached"
        partitions = partitioned_shard.rdd.glom().collect()
        assert len(partitions) == 3

    def test_unique(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        location_list = data_shard["location"].unique()
        assert len(location_list) == 6

    def test_split(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        trans_data_shard = data_shard.transform_shard(lambda df: (df[0:-1], df[-1:]))
        assert trans_data_shard.is_cached(), "trans_data_shard should be cached"
        shards_splits = trans_data_shard.split()
        assert not trans_data_shard.is_cached(), "shards_splits should be uncached"
        trans_data_shard.uncache()
        del trans_data_shard
        assert len(shards_splits) == 2
        assert shards_splits[0].is_cached(), "shards in shards_splits should be cached"
        data1 = shards_splits[0].collect()
        data2 = shards_splits[1].collect()
        assert len(data1[0].index) > 1
        assert len(data2[0].index) == 1

    def test_len(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        assert len(data_shard) == 14
        assert len(data_shard['ID']) == 14
        with self.assertRaises(Exception) as context:
            len(data_shard['abc'])
        self.assertTrue('Invalid key for this XShards' in str(context.exception))

        def to_dict(df):
            return {'ID': df['ID'].to_numpy(), 'location': df['location'].to_numpy()}
        data_shard = data_shard.transform_shard(to_dict)
        assert len(data_shard['ID']) == 14
        assert len(data_shard) == 4
        with self.assertRaises(Exception) as context:
            len(data_shard['abc'])
        self.assertTrue('Invalid key for this XShards' in str(context.exception))

        def to_number(d):
            return 4
        data_shard = data_shard.transform_shard(to_number)
        assert len(data_shard) == 2
        with self.assertRaises(Exception) as context:
            len(data_shard['abc'])
        self.assertTrue('No selection operation available for this XShards' in
                        str(context.exception))

    def test_save(self):
        temp = tempfile.mkdtemp()
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        path = os.path.join(temp, "data.pkl")
        data_shard.save_pickle(path)
        shards = zoo.orca.data.SparkXShards.load_pickle(path)
        assert isinstance(shards, zoo.orca.data.SparkXShards)
        shutil.rmtree(temp)

    def test_transform(self):
        def trans_func(df):
            data1 = {'ID': df['ID'].values, 'price': df['sale_price'].values}
            data2 = {'location': df['location'].values}
            return {'x': data1, 'y': data2}
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = zoo.orca.data.pandas.read_csv(file_path)
        assert data_shard.is_cached(), "data_shard should be cached"
        transformed_data_shard = data_shard.transform_shard(trans_func)
        assert not data_shard.is_cached(), "data_shard should be uncached"
        assert transformed_data_shard.is_cached(), "transformed_data_shard should be cached"
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"
        trans_data = transformed_data_shard.collect()
        assert len(trans_data) == 2, "number of shard should be 2"
        trans_dict = trans_data[0]
        assert "x" in trans_dict, "x is not in the dictionary"

    def test_transform_broadcast(self):
        def negative(df, column_name, minus_val):
            df[column_name] = df[column_name] * (-1)
            df[column_name] = df[column_name] - minus_val.value
            return df

        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data = data_shard.collect()
        assert data[0]["value"].values[0] > 0, "value should be positive"
        col_name = "value"
        minus_val = 2
        minus_val_shared_value = SharedValue(minus_val)
        trans_shard = data_shard.transform_shard(negative, col_name,
                                                 minus_val_shared_value)
        data2 = trans_shard.collect()
        assert data2[0]["value"].values[0] < 0, "value should be negative"
        assert data[0]["value"].values[0] + data2[0]["value"].values[0] == -2, "value should be -2"

    def test_get_item(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        data_shard = zoo.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
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

    def test_for_each(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        shards = zoo.orca.data.pandas.read_csv(file_path)

        def get_item(data, key):
            return data[key]
        result1 = shards._for_each(get_item, 'location')
        import pandas as pd
        assert isinstance(result1.first(), pd.Series)
        result2 = shards._for_each(get_item, 'abc')
        assert isinstance(result2.first(), KeyError)


if __name__ == "__main__":
    pytest.main([__file__])

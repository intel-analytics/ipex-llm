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
from bigdl.orca.data.transformer import *
from bigdl.orca import init_orca_context, stop_orca_context

class TestSparkXShards(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")

    def test_repartition(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        # data_shard = bigdl.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data_shard = bigdl.orca.data.pandas.read_json(file_path)
        partitions_num_1 = data_shard.rdd.getNumPartitions()
        assert partitions_num_1 == 2, "number of partition should be 2"
        data_shard.cache()
        partitioned_shard = data_shard.repartition(1)

        # Test dict of numpy
        def to_numpy_dict(df):
            d = df.to_dict()
            return {k: np.array([v for v in d[k].values()]) for k in d.keys()}

        numpy_dict_shard = data_shard.transform_shard(to_numpy_dict)
        partitioned_numpy_dict_shard = numpy_dict_shard.repartition(1)
        assert partitioned_numpy_dict_shard.num_partitions() == 1, "number of partition should be 1"
        assert len(partitioned_numpy_dict_shard.collect()) == 1

        partitioned_numpy_dict_shard2 = numpy_dict_shard.repartition(3)
        assert partitioned_numpy_dict_shard2.num_partitions() == 3, \
            "number of partition should be 3"
        assert data_shard.is_cached(), "data_shard should be cached"
        assert partitioned_shard.is_cached(), "partitioned_shard should be cached"
        data_shard.uncache()
        assert not data_shard.is_cached(), "data_shard should be uncached"
        partitions_num_2 = partitioned_shard.rdd.getNumPartitions()
        assert partitions_num_2 == 1, "number of partition should be 1"

    def test_apply(self):
        file_path = os.path.join(self.resource_path, "orca/data/json")
        # data_shard = bigdl.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data_shard = bigdl.orca.data.pandas.read_json(file_path)
        data = data_shard.collect()
        assert data[0]["value"].values[0] > 0, "value should be positive"

        def negative(df, column_name):
            df[column_name] = df[column_name] * (-1)
            return df

        trans_data_shard = data_shard.transform_shard(negative, "value")
        data2 = trans_data_shard.collect()
        assert data2[0]["value"].values[0] < 0, "value should be negative"

    def test_partition_by_single_column(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        partitioned_shard = data_shard.partition_by(cols="location", num_partitions=4)
        partitions = partitioned_shard.rdd.glom().collect()
        assert len(partitions) == 4

        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        partitioned_shard = data_shard.partition_by(cols="location", num_partitions=3)
        assert not data_shard.is_cached(), "data_shard should be uncached"
        assert partitioned_shard.is_cached(), "partitioned_shard should be cached"
        partitions = partitioned_shard.rdd.glom().collect()
        assert len(partitions) == 3

    def test_unique(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        location_list = data_shard["location"].unique()
        assert len(location_list) == 6

    def test_split(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
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
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
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

    def test_transform(self):
        def trans_func(df):
            data1 = {'ID': df['ID'].values, 'price': df['sale_price'].values}
            data2 = {'location': df['location'].values}
            return {'x': data1, 'y': data2}
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
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
        # data_shard = bigdl.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data_shard = bigdl.orca.data.pandas.read_json(file_path)
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

    def test_for_each(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        shards = bigdl.orca.data.pandas.read_csv(file_path)

        def get_item(data, key):
            return data[key]
        result1 = shards._for_each(get_item, 'location')
        import pandas as pd
        assert isinstance(result1.first(), pd.Series)
        result2 = shards._for_each(get_item, 'abc')
        assert isinstance(result2.first(), KeyError)

    def test_zip(self):
        def negative(df, column_name, minus_val):
            df[column_name] = df[column_name] * (-1)
            df[column_name] = df[column_name] - minus_val
            return df

        file_path = os.path.join(self.resource_path, "orca/data/json")
        # data_shard = bigdl.orca.data.pandas.read_json(file_path, orient='columns', lines=True)
        data_shard = bigdl.orca.data.pandas.read_json(file_path)
        data_shard = data_shard.repartition(2)
        data_shard.cache()
        print(data_shard.collect()[0])
        transformed_shard = data_shard.transform_shard(negative, "value", 2)
        zipped_shard = data_shard.zip(transformed_shard)
        assert not transformed_shard.is_cached(), "transformed_shard should be uncached."
        data = zipped_shard.collect()
        assert data[0][0]["value"].values[0] + data[0][1]["value"].values[0] == -2, \
            "value should be -2"
        list1 = list([1, 2, 3])
        with self.assertRaises(Exception) as context:
            data_shard.zip(list1)
        self.assertTrue('other should be a SparkXShards' in str(context.exception))
        transformed_shard = transformed_shard.repartition(data_shard.num_partitions() - 1)
        with self.assertRaises(Exception) as context:
            data_shard.zip(transformed_shard)
        self.assertTrue('The two SparkXShards should have the same number of partitions' in
                        str(context.exception))
        dict_data = [{"x": 1, "y": 2}, {"x": 2, "y": 3}]
        sc = init_nncontext()
        rdd = sc.parallelize(dict_data)
        dict_shard = SparkXShards(rdd)
        dict_shard = dict_shard.repartition(1)
        with self.assertRaises(Exception) as context:
            transformed_shard.zip(dict_shard)
        self.assertTrue('The two SparkXShards should have the same number of elements in '
                        'each partition' in str(context.exception))

    def test_transform_with_repartition(self):
        # shards of pandas dataframe
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        partitions = data_shard.rdd.glom().collect()
        for par in partitions:
            assert len(par) <= 1

        def negative(df, column_name):
            df[column_name] = df[column_name] * (-1)
            return df
        shard2 = data_shard.transform_shard(negative, "sale_price")

        shard3 = shard2.repartition(4)
        partitions3 = shard3.rdd.glom().collect()
        for par in partitions3:
            assert len(par) <= 1

        shard4 = shard2.repartition(1)
        partitions4 = shard4.rdd.glom().collect()
        for par in partitions4:
            assert len(par) <= 1

        shard5 = shard4.transform_shard(negative, "sale_price")
        partitions5 = shard5.rdd.glom().collect()
        for par in partitions5:
            assert len(par) <= 1
        # shards of list
        data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        sc = init_nncontext()
        rdd = sc.parallelize(data)
        data_shard = SparkXShards(rdd)
        shard2 = data_shard.repartition(6)
        partitions2 = shard2.rdd.glom().collect()
        for par in partitions2:
            assert len(par) <= 1
        shard3 = data_shard.repartition(1)
        partitions2 = shard3.rdd.glom().collect()
        for par in partitions2:
            assert len(par) <= 1

        # shards of numpy array
        data = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                np.array([9, 10, 11, 12]), np.array([13, 14, 15, 16])]
        sc = init_nncontext()
        rdd = sc.parallelize(data)
        data_shard = SparkXShards(rdd)
        shard2 = data_shard.repartition(6)
        partitions2 = shard2.rdd.glom().collect()
        for par in partitions2:
            assert len(par) <= 1
        shard3 = data_shard.repartition(1)
        partitions2 = shard3.rdd.glom().collect()
        for par in partitions2:
            assert len(par) <= 1

    def test_to_spark_df(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, header=0, names=['user', 'item'],
                                                   usecols=[0, 1])
        df = data_shard.to_spark_df()
        df.show()

    def test_spark_df_to_shards(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[1]")\
            .appName('test_spark_backend')\
            .config("spark.driver.memory", "6g").getOrCreate()
        df = spark.read.csv(file_path)
        data_shards = spark_df_to_pd_sparkxshards(df)

    def test_minmaxscale_shards(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        scale = MinMaxScaler(inputCol=["sale_price"], outputCol="sale_price_scaled")
        transformed_data_shard = scale.fit_transform(data_shard)

    def test_standardscale_shards(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")

        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        scale = StandardScaler(inputCol="sale_price", outputCol="sale_price_scaled")
        transformed_data_shard = scale.fit_transform(data_shard)

    def test_max_values(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv/morgage1.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        max_value = data_shard.max_values('sale_price')
        assert max_value == 475000, "max value of sale_price should be 2"

    def test_merge_shards(self):
        from bigdl.orca.data.utils import spark_df_to_pd_sparkxshards
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df1 = spark.createDataFrame([
            (1, 2.),
            (2, 3.),
            (3, 5.),
            (4, 1.)
            ], schema=['a', 'b'])
        df2 = spark.createDataFrame([
            (1, 7),
            (2, 8),
            (4, 9),
            (5, 9)
            ], schema=['a', 'c'])
        data_shard1 = spark_df_to_pd_sparkxshards(df1)
        data_shard2 = spark_df_to_pd_sparkxshards(df2)
        merged_shard = data_shard1.merge(data_shard2, on='a')
        merged_shard_df = merged_shard.to_spark_df()
        assert len(merged_shard)==3 and merged_shard_df.columns==['a','b','c']

    def test_usecols(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, usecols=[0, 1])
        data = data_shard.collect()
        df = data[0]
        assert "sale_price" in df.columns
        assert "location" not in df.columns
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, usecols=["ID"])
        data = data_shard.collect()
        df2 = data[0]
        assert "ID" in df2.columns and "location" not in df2.columns

        def filter_col(name):
            return name == "sale_price"

        data_shard = bigdl.orca.data.pandas.read_csv(file_path, usecols=filter_col)
        data = data_shard.collect()
        df3 = data[0]
        assert "sale_price" in df3.columns and "location" not in df3.columns

    def test_dtype(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, dtype="float")
        data = data_shard.collect()
        df = data[0]
        assert df.location.dtype == "float64"
        assert df.ID.dtype == "float64"
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, dtype={"sale_price": np.float32, "ID": np.int64})
        data = data_shard.collect()
        df2 = data[0]
        assert df2.sale_price.dtype == "float32" and df2.ID.dtype == "int64"

    def test_squeeze(self):
        import pandas as pd
        file_path = os.path.join(self.resource_path, "orca/data/single_column.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, squeeze=True)
        data = data_shard.collect()
        df = data[0]
        assert isinstance(df, pd.Series)

    def test_index_col(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv/morgage1.csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, index_col="ID")
        data = data_shard.collect()
        df = data[0]
        assert 100529 in df.index

    def test_mix(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, header=0, names=['user', 'item'],
                                                   usecols=[0, 1])
        data = data_shard.collect()
        df = data[0]
        assert "user" in df.columns
        assert "item" in df.columns
        with self.assertRaises(Exception) as context:
            data_shard = bigdl.orca.data.pandas.read_csv(file_path, header=0,
                                                       names=['ID', 'location'], usecols=["ID"])
            data = data_shard.collect()
        self.assertTrue('Passed names did not match usecols'
                        in str(context.exception))
        data_shard = bigdl.orca.data.pandas.read_csv(file_path, header=0,
                                                   names=['user', 'item'], usecols=[0, 1],
                                                   dtype={0: np.float32, 1: np.int32})
        data = data_shard.collect()
        df2 = data[0]
        assert df2.user.dtype == "float32" and df2.item.dtype == "int32"

        data_shard = bigdl.orca.data.pandas.read_csv(file_path, header=0,
                                                   names=['user', 'item', 'location'],
                                                   usecols=[1, 2])
        data = data_shard.collect()
        df2 = data[0]
        assert "user" not in df2.columns
        assert "item" in df2.columns
        assert "location" in df2.columns

        data_shard = bigdl.orca.data.pandas.read_csv(file_path, header=0,
                                                   names=['user', 'item', 'rating'],
                                                   usecols=['user', 'item'],
                                                   dtype={0: np.float32, 1: np.int32})
        data = data_shard.collect()
        df2 = data[0]
        assert df2.user.dtype == "float32" and df2.item.dtype == "int32"

        with self.assertRaises(Exception) as context:
            data_shard = bigdl.orca.data.pandas.read_csv(file_path, header=0,
                                                       names=['user', 'item'], usecols=[0, 1],
                                                       dtype={1: np.float32, 2: np.int32})
            data = data_shard.collect()
            print(len(data))
        self.assertTrue('column index to be set type is not in current dataframe'
                        in str(context.exception))

    def test_select(self):
        import pandas as pd
        df1 = {'id': [1, 2, 3, 4, 5],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df1 = pd.DataFrame(df1, columns=['id', 'created_at', 'type'])
        df2 = {'id': [6, 7, 8, 9, 10],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df2 = pd.DataFrame(df2, columns=['id', 'created_at', 'type'])
        sc = get_spark_context()
        shard = SparkXShards(sc.parallelize([df1, df2]))
        selected = shard.select("id").collect()[0]
        assert (list(selected.columns) == ["id"])

    def test_concat_to_pdf(self):
        import pandas as pd
        df1 = {'id': [1, 2, 3, 4, 5],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df1 = pd.DataFrame(df1, columns=['id', 'created_at', 'type'])
        df2 = {'id': [6, 7, 8, 9, 10],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df2 = pd.DataFrame(df2, columns=['id', 'created_at', 'type'])
        sc = get_spark_context()
        shard = SparkXShards(sc.parallelize([df1, df2]))
        concated = shard.concat_to_pdf()
        assert(len(concated) == 10)

    def test_sample(self):
        import pandas as pd
        df1 = {'id': [1, 2, 3, 4, 5],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df1 = pd.DataFrame(df1, columns=['id', 'created_at', 'type'])
        df2 = {'id': [6, 7, 8, 9, 10],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df2 = pd.DataFrame(df2, columns=['id', 'created_at', 'type'])
        sc = get_spark_context()
        shard = SparkXShards(sc.parallelize([df1, df2]))
        df1 = shard.collect()[0]
        print(len(df1))
        sampled = shard.sample(frac=0.4)
        assert(len(sampled) == 4)

    def test_describe(self):
        import pandas as pd
        df1 = {'id': [1, 2, 3, 4, 5],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df1 = pd.DataFrame(df1, columns=['id', 'created_at', 'type'])
        df2 = {'id': [6, 7, 8, 9, 10],
              'created_at': ['2020-02-01', '2020-02-02', '2020-02-02', '2020-02-02', '2020-02-03'],
              'type': ['red', None, 'blue', 'blue', 'yellow']}
        df2 = pd.DataFrame(df2, columns=['id', 'created_at', 'type'])
        sc = init_orca_context(cores=2)
        rdd = sc.parallelize([df1, df2], numSlices=2)
        shard = SparkXShards(rdd)
        description1 = shard.describe()
        description = shard.describe("id")
        print(description)
        columns = list(shard.get_schema()['columns'])
        print("columns", columns)
        assert (description is not None)
        assert (description1 is not None)

    def test_lazy_xshards(self):
        file_path = os.path.join(self.resource_path, "orca/data/csv")
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        assert data_shard.is_cached()

        def increment(df):
            df["ID"] = df["ID"] + 1
            return df

        data_shard1 = data_shard.transform_shard(increment)
        assert not data_shard.is_cached()
        assert data_shard1.is_cached()
        data_shard2 = data_shard1.repartition(data_shard1.num_partitions()*2)
        assert not data_shard1.is_cached()
        assert data_shard2.is_cached()

        lazy_shard = data_shard2.to_lazy()
        lazy_shard1 = lazy_shard.transform_shard(increment)
        assert lazy_shard.is_cached()  # Same as data_shard2
        assert not lazy_shard1.is_cached()
        lazy_shard2 = lazy_shard1.repartition(lazy_shard1.num_partitions()//2)
        assert not lazy_shard2.is_cached()
        assert data_shard2.is_cached()

if __name__ == "__main__":
    pytest.main([__file__])

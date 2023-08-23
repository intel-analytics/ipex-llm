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
from py4j.protocol import Py4JError

from bigdl.orca.data.utils import *
from bigdl.orca import OrcaContext
from bigdl.dllib.utils.common import (get_node_and_core_number,
                                      get_spark_sql_context,
                                      get_spark_context)
from bigdl.dllib.utils import nest
from bigdl.dllib.utils.log4Error import invalidInputError

import numpy as np

from typing import TYPE_CHECKING, Any
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union
)

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas.core.frame import DataFrame as PandasDataFrame
    from pyspark.rdd import PipelinedRDD, RDD
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame
    from pyspark.sql.column import Column
    from ray.data.dataset import Dataset


class XShards(object):
    """

    A collection of data which can be pre-processed in parallel.
    """
    def transform_shard(self, func: Callable, *args):
        """

        Transform each shard in the XShards using specified function.

        :param func: pre-processing function
        :param args: arguments for the pre-processing function
        :return: DataShard
        """
        pass

    def collect(self):
        """

        Returns a list that contains all of the elements in this XShards

        :return: list of elements
        """
        pass

    def num_partitions(self):
        """

        return the number of partitions in this XShards

        :return: an int
        """
        pass

    @classmethod
    def load_pickle(cls, path: str, minPartitions: Optional[int] = None) -> "SparkXShards":
        """

        Load XShards from pickle files.

        :param path: The pickle file path/directory
        :param minPartitions: The minimum partitions for the XShards
        :return: SparkXShards object
        """
        sc = OrcaContext.get_spark_context()
        return SparkXShards(sc.pickleFile(path, minPartitions))

    @staticmethod
    def partition(
        data: Union["ndarray",
                    List["ndarray"],
                    Tuple["ndarray", "ndarray"],
                    Dict[str, Union["ndarray", Tuple["ndarray"], List["ndarray"]]]],
        num_shards: Optional[int] = None
    ) -> "SparkXShards":
        """

        Partition local in memory data and form a SparkXShards

        :param data: np.ndarray, a tuple, list, dict of np.ndarray, or a nested structure
        made of tuple, list, dict with ndarray as the leaf value
        :param num_shards: the number of shards that the data will be partitioned into
        :return: a SparkXShards
        """
        sc = OrcaContext.get_spark_context()
        node_num, core_num = get_node_and_core_number()
        shard_num = node_num * core_num if num_shards is None else num_shards
        import numpy as np
        type_err_msg = """
The types supported in bigdl.orca.data.XShards.partition are
1. np.ndarray
2. a tuple, list, dict of np.ndarray
3. nested structure made of tuple, list, dict with ndarray as the leaf value

But got data of type {}
        """.format(type(data))
        supported_types = {list, tuple, dict}
        if isinstance(data, np.ndarray):
            if data.shape[0] < shard_num:
                invalidInputError(False,
                                  "The length of data {} is smaller than the total number "
                                  "of shards {}. Please adjust the num_shards option to be "
                                  "at most {}.".format(data.shape[0], shard_num, data.shape[0]))
            arrays = np.array_split(data, shard_num)
            rdd = sc.parallelize(arrays)
        else:
            invalidInputError(type(data) in supported_types, type_err_msg)
            flattened = nest.flatten(data)
            data_length = len(flattened[0])
            data_to_be_shard = []  # type:ignore
            if data_length < shard_num:
                invalidInputError(False,
                                  "The length of data {} is smaller than the total number "
                                  "of shards {}. Please adjust the num_shards option to be "
                                  "at most {}.".format(data_length, shard_num, data_length))
            for i in range(shard_num):
                data_to_be_shard.append([])
            for x in flattened:
                invalidInputError(len(x) == data_length,
                                  "the ndarrays in data must all have the same size in first"
                                  " dimension, got first ndarray of size {} and"
                                  " another {}".format(data_length, len(x)))
                x_parts = np.array_split(x, shard_num)
                for idx, x_part in enumerate(x_parts):
                    data_to_be_shard[idx].append(x_part)

            data_to_be_shard = [nest.pack_sequence_as(data, shard) for shard in data_to_be_shard]
            rdd = sc.parallelize(data_to_be_shard)

        data_shards = SparkXShards(rdd)
        return data_shards


class SparkXShards(XShards):
    """
    A collection of data which can be pre-processed in parallel on Spark.
    """
    def __init__(self,
                 rdd: Union["PipelinedRDD", "RDD"],
                 transient: bool = False,
                 class_name: Optional[str] = None) -> None:
        self.rdd = rdd
        self.user_caching = self.rdd.is_cached
        if transient:
            self.eager = False
        else:
            self.eager = OrcaContext._eager_mode
            self.rdd.cache()
        if self.eager:
            self.compute()
            self.is_lazy = False
        else:
            self.is_lazy = True
        self.type = {}
        if class_name:
            self.type['class_name'] = class_name

    @classmethod
    def lazy(cls,
             rdd: Union["PipelinedRDD", "RDD"],
             class_name: str = None) -> "SparkXShards":
        """
        Create an instance of SparkXShards that computes lazily.
        """
        return SparkXShards(rdd, transient=True, class_name=class_name).to_lazy()

    def _create(self,
                rdd: Union["PipelinedRDD", "RDD"],
                class_name: str = None) -> "SparkXShards":
        """
        Create an instance of SparkXShards after transformation.
        """
        if self.is_lazy:
            return SparkXShards(rdd, transient=True, class_name=class_name).to_lazy()
        else:
            return SparkXShards(rdd, class_name=class_name)

    def to_lazy(self) -> "SparkXShards":
        """
        Making the current SparkXShards lazy won't change the behavior of the current SparkXShards.
        Cached data won't be uncached unless uncache is explicitly invoked.
        After being marked as lazy, future operations would be performed lazily.

        :return: An instance of SparkXShards that computes lazily.
        """
        self.eager = False
        self.is_lazy = True
        self.user_caching = True
        return self

    def transform_shard(self, func: Callable, *args) -> "SparkXShards":
        """

        Return a new SparkXShards by applying a function to each shard of this SparkXShards

        :param func: python function to process data. The first argument is the data shard.
        :param args: other arguments in this function.
        :return: a new SparkXShards.
        """
        def transform(iter, func, *args):
            for x in iter:
                yield func(x, *args)

        transformed_shard = self._create(self.rdd.mapPartitions(lambda iter:
                                                                transform(iter, func, *args)))
        self._uncache()
        return transformed_shard

    def collect(self) -> List[Any]:
        """

        Returns a list that contains all of the elements in this SparkXShards

        :return: a list of data elements.
        """
        return self.rdd.collect()

    def first(self):
        """
        Returns the first element in the rdd of SparkXShards
        :return: a record of data.
        """
        return self.rdd.first()

    def take(self, n) -> List[Any]:
        """
        Returns n element in the rdd of SparkXShards
        :return: n records of data.
        """
        return self.rdd.take(n)

    def cache(self) -> "SparkXShards":
        """

        Persist this SparkXShards in memory

        :return:
        """
        self.user_caching = True
        self.rdd.cache()
        return self

    def uncache(self) -> "SparkXShards":
        """

        Make this SparkXShards as non-persistent, and remove all blocks for it from memory

        :return:
        """
        self.user_caching = False
        try:
            if self.is_cached():
                self.rdd.unpersist()
        except (Py4JError, TypeError):
            pass
        return self

    def _uncache(self) -> None:
        if not self.user_caching:
            self.uncache()

    def is_cached(self) -> bool:
        return self.rdd.is_cached

    def compute(self) -> "SparkXShards":
        self.rdd.count()
        return self

    def num_partitions(self) -> int:
        """

        Get number of partitions for this SparkXShards.

        :return: number of partitions.
        """
        return self.rdd.getNumPartitions()

    def repartition(self, num_partitions: int) -> "SparkXShards":
        """

        Return a new SparkXShards that has exactly num_partitions partitions.

        :param num_partitions: target number of partitions
        :return: a new SparkXShards object.
        """
        class_name = self._get_class_name()
        if class_name == 'pandas.core.frame.DataFrame':
            import pandas as pd

            rdd = self.rdd \
                .flatMap(lambda df: df.apply(lambda row: (row[0], row.values.tolist()), axis=1)
                         .values.tolist()) \
                .partitionBy(num_partitions)

            schema = self.get_schema()

            def merge_rows(iter):
                data = [value[1] for value in list(iter)]
                if data:
                    df = pd.DataFrame(data=data, columns=schema['columns']) \
                        .astype(schema['dtypes'])
                    return [df]
                else:
                    # no data in this partition
                    return iter

            repartitioned_shard = self._create(rdd.mapPartitions(merge_rows),
                                               class_name=class_name)
        elif class_name == 'builtins.list':
            if num_partitions > self.rdd.getNumPartitions():
                rdd = self.rdd \
                    .flatMap(lambda data: data) \
                    .repartition(num_partitions)

                repartitioned_shard = self._create(rdd.mapPartitions(
                    lambda iter: [list(iter)]), class_name=class_name)
            else:
                rdd = self.rdd.coalesce(num_partitions)
                from functools import reduce
                repartitioned_shard = self._create(rdd.mapPartitions(
                    lambda iter: [reduce(lambda l1, l2: l1 + l2, iter)]),  # type:ignore
                    class_name=class_name)  # type:ignore
        elif class_name == 'numpy.ndarray':
            elem = self.rdd.first()
            shape = elem.shape
            dtype = elem.dtype
            if len(shape) > 0:
                if num_partitions > self.rdd.getNumPartitions():
                    rdd = self.rdd \
                        .flatMap(lambda data: list(data)) \
                        .repartition(num_partitions)

                    repartitioned_shard = self._create(rdd.mapPartitions(
                        lambda iter: np.stack([list(iter)], axis=0).astype(dtype)),
                        class_name=class_name)
                else:
                    rdd = self.rdd.coalesce(num_partitions)
                    from functools import reduce
                    repartitioned_shard = self._create(rdd.mapPartitions(
                        lambda iter: [np.concatenate(list(iter), axis=0)]),
                        class_name=class_name)
            else:
                repartitioned_shard = self._create(self.rdd.repartition(num_partitions),
                                                   class_name=class_name)
        elif class_name == "builtins.dict":
            elem = self.rdd.first()
            keys = list(elem.keys())
            dtypes = []
            dict_of_batched_ndarray = True
            # Check if all values are ndarray and shape > 1
            for v in elem.values():
                if v.__class__.__name__ != "ndarray" or len(v.shape) == 0:
                    dict_of_batched_ndarray = False
                    break
                else:
                    dtypes.append(v.dtype)
            if dict_of_batched_ndarray:
                def dict_to_unbatched_list(d):
                    values = [list(d[k]) for k in keys]
                    return list(zip(*values))

                def to_batched_dict(iter):
                    batch_values = list(zip(*iter))
                    if not batch_values:
                        return []
                    batch_ndarrays = [np.stack(v, axis=0).astype(dtype)
                                      for v, dtype in zip(batch_values, dtypes)]
                    return [dict(zip(keys, batch_ndarrays))]

                # If number of records in a partition <= 10, may produce empty partition
                rdd = self.rdd.flatMap(lambda data: dict_to_unbatched_list(data)) \
                    .repartition(num_partitions)
                repartitioned_shard = self._create(rdd.mapPartitions(
                    lambda iter: to_batched_dict(iter)), class_name=class_name)
            else:
                repartitioned_shard = self._create(self.rdd.repartition(num_partitions),
                                                   class_name=class_name)
        else:
            repartitioned_shard = self._create(self.rdd.repartition(num_partitions),
                                               class_name=class_name)
        self._uncache()
        return repartitioned_shard

    def partition_by(self, cols: str,
                     num_partitions: Optional[int] = None) -> Optional["SparkXShards"]:
        """

        Return a new SparkXShards partitioned using the specified columns.
        This is only applicable for SparkXShards of Pandas DataFrame.

        :param cols: specified columns to partition by.
        :param num_partitions: target number of partitions. If not specified,
        the new SparkXShards would keep the current partition number.
        :return: a new SparkXShards.
        """
        if self._get_class_name() == 'pandas.core.frame.DataFrame':
            import pandas as pd
            schema = self.get_schema()
            # if partition by a column
            if isinstance(cols, str):
                if not isinstance(schema, Dict) or cols not in schema['columns']:
                    invalidInputError(False,
                                      "The partition column is not in the DataFrame")
                # change data to key value pairs
                rdd = self.rdd.flatMap(
                    lambda df: df.apply(
                        lambda row: (row[cols], row.values.tolist()), axis=1).values.tolist())

                partition_num = self.rdd.getNumPartitions() if not num_partitions \
                    else num_partitions
                # partition with key
                partitioned_rdd = rdd.partitionBy(partition_num)
            else:
                invalidInputError(False,
                                  "Only support partition by a column name")

            def merge(iterator):
                data = [value[1] for value in list(iterator)]
                if data:
                    df = pd.DataFrame(data=data, columns=schema['columns']).astype(schema['dtypes'])
                    return [df]
                else:
                    # no data in this partition
                    return []

            # merge records to df in each partition
            partitioned_shard = SparkXShards(partitioned_rdd.mapPartitions(merge))
            self._uncache()
            return partitioned_shard
        else:
            invalidInputError(False,
                              "Currently only support partition by for XShards"
                              " of Pandas DataFrame")
        return None

    def unique(self) -> Optional["ndarray"]:
        """

        Return a unique list of elements of this SparkXShards.
        This is only applicable for SparkXShards of Pandas Series.

        :return: a unique list of elements of this SparkXShards.
        """
        if self._get_class_name() == 'pandas.core.series.Series':
            import pandas as pd
            rdd = self.rdd.map(lambda s: s.unique())
            import numpy as np
            result = rdd.reduce(lambda list1, list2: pd.unique(np.concatenate((list1, list2),
                                                                              axis=0)))
            return result
        else:
            invalidInputError(False,
                              "Currently only support unique() on XShards of Pandas Series")
        return None

    def deduplicates(self) -> Optional["SparkXShards"]:
        if self._get_class_name() == 'pandas.core.frame.DataFrame':
            import pandas as pd
            df = self.to_spark_df()
            distinctDF = df.distinct()
            data_shards = spark_df_to_pd_sparkxshards(distinctDF)
            return data_shards
        else:
            invalidInputError(False,
                              "Currently only support dedup() on XShards of Pandas DataFrame")
        return None

    def sort_values(self, col_names: Optional[Union[str, List[str]]]=None,
                    ascending: bool = True) -> Optional["SparkXShards"]:
        """
        Sort the value of shards. This is only applicable for SparkXShards of Pandas Series.

        :param col_names list of column or column names to sort by
        :param ascending bool, default True. Specify sort orders
        :return: a new SparkXShards sorted by the specified columns.
        """
        if self._get_class_name() == 'pandas.core.frame.DataFrame':
            import pandas as pd
            df = self.to_spark_df()
            sqlContext = get_spark_sql_context(get_spark_context())
            defaultPartitionNum = sqlContext.getConf("spark.sql.shuffle.partitions")
            partitionNum = df.rdd.getNumPartitions()
            sqlContext.setConf("spark.sql.shuffle.partitions", str(partitionNum))
            sort_df = df.sort(col_names, ascending=ascending)  # type:ignore
            data_shards = spark_df_to_pd_sparkxshards(sort_df)
            sqlContext.setConf("spark.sql.shuffle.partitions", defaultPartitionNum)
            return data_shards
        else:
            invalidInputError(False,
                              "Currently only support sort() on XShards of Pandas DataFrame")
        return None

    def max_values(self, col_name: str) -> Union[int, float, None]:
        """
        Get the max values of the column name. This is only applicable for SparkXShards
         of Pandas Series.

        :param col_name column name that need return the max value
        :return: maximum value for the specified columns.
        """
        if self._get_class_name() == 'pandas.core.frame.DataFrame':
            import pandas as pd
            rdd = self.rdd.map(lambda s: s[col_name].max())
            max_value = rdd.reduce(lambda value1, value2: max(value1, value2))
            return max_value
        else:
            invalidInputError(False,
                              "Currently only support max() on XShards of Pandas DataFrame")
        return None

    def get_null_sum(self) -> "PandasDataFrame":
        """
        With SparkXShards of pandas data frame, the api will get null numbers for
        each column. For other type of SparkXShards, it will throw exception

       :return: pandas data frame with 2 columns, `col` represents column name,
        `total` represents null numbers
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support get_null_sum() on"
                              " XShards of Pandas DataFrame")

        def get_na_sum(iter):
            for df in iter:
                import pandas as pd
                series = df.isnull().sum()
                df = pd.DataFrame({'col': series.index, 'total': series.values})
                return [df]

        na_cnt_rdd = self.rdd.mapPartitions(get_na_sum)

        na_cnt = na_cnt_rdd.reduce(lambda l1, l2: l1.add(l2))
        return na_cnt

    def drop_missing_value(self) -> "SparkXShards":
        """
        With SparkXShards of pandas data frame, the api will drop null values in shards.
         For other type of SparkXShards, it will throw exception

       :return: a new SparkXShards without null values
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support assembleFeatureLabelCols() on"
                              " XShards of Pandas DataFrame")

        null_cnt = self.get_null_sum()

        def drop_missing_data(pdf):
            df2 = pdf.drop((null_cnt[null_cnt['total'] > 0]['col']), 1)
            return df2

        # dealing with missing data
        return self.transform_shard(drop_missing_data)

    def assembleFeatureLabelCols(self,
                                 featureCols: List[Union[str, "Column"]],
                                 labelCols: List[Union[str, "Column"]]) -> "SparkXShards":
        """
        The api is used to merge/convert one or multiple feature columns into a numpy array,
        merge/convert one or multiple label columns into a numpy array.

        :param featureCols: a list of feature columns.
        :param labelCols: a list of label columns.
        :return: SparkXShards of dictionary, key is assembled feature numpy array, value is
         assembled label numpy array

        eg:
        shards: SparkXShards of pandas data frame with 9 cols ['f1', 'f2', 'f3', 'f4', 'f5', 'f6',
         'f7', 'f8', 'lable']
            f1   f2  f3  f4   f5    f6     f7  f8  label
             6  148  72  35    0  33.6  0.627  50      1
             1   85  66  29    0  26.6  0.351  31      0
             8  183  64   0    0  23.3  0.672  32      1
             1   89  66  23   94  28.1  0.167  21      0
             0  137  40  35  168  43.1  2.288  33      1

        transform_shards =
          shards.assembleFeatureLabelCols(featureCols=['f1', 'f2', 'f3', 'f4', 'f5', 'f6',
           'f7', 'f8'], labelCols=['label'])

        transform_shards will be SparkXShards of dictionary. key will be a stacked numpy array
        (stack feature columns), value will be a numpy array
        {'x': array([[  6.   , 148.   ,  72.   , ...,  33.6  ,   0.627,  50.   ],
           [  1.   ,  85.   ,  66.   , ...,  26.6  ,   0.351,  31.   ],
           [  8.   , 183.   ,  64.   , ...,  23.3  ,   0.672,  32.   ],
           [  1.   , 89.   ,  66.   , ...,  28.1  ,   0.167,  21.   ],
           [  0.   , 137.   ,  40.   , ...,  43.1  ,  2.288, 33.   ]]),
         'y': array([[1],
           [0],
           [1],
           [0],
           [1]
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support assembleFeatureLabelCols() on"
                              " XShards of Pandas DataFrame")

        def to_shard_dict(df):
            featureLists = [df[feature_col].to_numpy() for feature_col in featureCols]
            labelLists = [df[label_col].to_numpy() for label_col in labelCols]
            result = {
                "x": np.stack(featureLists, axis=1),
                "y": np.stack(labelLists, axis=1)}
            return result

        invalidInputError(type(featureCols) == list, "expect featureCols is a list")
        invalidInputError(type(labelCols) == list, "expect labelCols is a list")
        transformed_shard = self.transform_shard(to_shard_dict)
        return transformed_shard

    def split(self) -> Optional[List["SparkXShards"]]:
        """

        Split SparkXShards into multiple SparkXShards.
        Each element in the SparkXShards needs be a list or tuple with same length.

        :return: Splits of SparkXShards. If element in the input SparkDataShard is not
                list or tuple, return list of input SparkDataShards.
        """
        # get number of splits
        list_split_length = self.rdd.map(
            lambda data: len(data) if isinstance(data, list) or isinstance(data, tuple) else 1)\
            .collect()
        # check if each element has same splits
        if list_split_length.count(list_split_length[0]) != len(list_split_length):
            invalidInputError(False,
                              "Cannot split this XShards because its partitions "
                              "have different split length")
        else:
            if list_split_length[0] > 1:
                def get_data(order):
                    def transform(data):
                        return data[order]

                    return transform

                split_shard_list = [SparkXShards(self.rdd.map(get_data(i)))
                                    for i in range(list_split_length[0])]
                self._uncache()
                return split_shard_list
            else:
                return [self]
        return None

    def zip(self, other: "SparkXShards") -> Optional["SparkXShards"]:
        """

        Zips this SparkXShards with another one, returning key-value pairs with the first element
        in each SparkXShards, second element in each SparkXShards, etc. Assumes that the two
        SparkXShards have the *same number of partitions* and the *same number of elements
        in each partition*(e.g. one was made through a transform_shard on the other

        :param other: another SparkXShards
        :return: zipped SparkXShards
        """
        invalidInputError(isinstance(other, SparkXShards), "other should be a SparkXShards")
        invalidInputError(self.num_partitions() == other.num_partitions(),
                          "The two SparkXShards should have the same number of partitions")
        try:
            rdd = self.rdd.zip(other.rdd)
            zipped_shard = SparkXShards(rdd)
            other._uncache()
            self._uncache()
            return zipped_shard
        except Exception:
            invalidInputError(False,
                              "The two SparkXShards should have the same number of elements "
                              "in each partition")
        return None

    def group_by(
        self,
        columns: Union[str, List[str]]=[],
        agg: Union[Dict[str, List[str]], List[str], Dict[str, str], str]="count",
        join: bool = False
    ) -> "SparkXShards":
        """
        Group the Shards with specified columns and then run aggregation. Optionally join the
        result with the original Shards.

        :param columns: str or a list of str. Columns to group the SparkXShards. If it is an
               empty list, aggregation is run directly without grouping. Default is [].
        :param agg: str, list or dict. Aggregate functions to be applied to grouped SparkXShards.
               Default is "count".
               Supported aggregate functions are: "max", "min", "count", "sum", "avg", "mean",
               "sumDistinct", "stddev", "stddev_pop", "variance", "var_pop", "skewness", "kurtosis",
               "collect_list", "collect_set", "approx_count_distinct", "first", "last".
               If agg is a str, then agg is the aggregate function and the aggregation is performed
               on all columns that are not in `columns`.
               If agg is a list of str, then agg is a list of aggregate function and the aggregation
               is performed on all columns that are not in `columns`.
               If agg is a single dict mapping from str to str, then the key is the column
               to perform aggregation on, and the value is the aggregate function.
               If agg is a single dict mapping from str to list, then the key is the
               column to perform aggregation on, and the value is list of aggregate functions.

               Examples:
               agg="sum"
               agg=["last", "stddev"]
               agg={"*":"count"}
               agg={"col_1":"sum", "col_2":["count", "mean"]}
        :param join: boolean. If True, join the aggregation result with original SparkXShards.

        :return: A new SparkXShards with aggregated column fields.
        """

        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support sort() on XShards of Pandas DataFrame")

        df = self.to_spark_df()
        sqlContext = get_spark_sql_context(get_spark_context())
        defaultPartitionNum = sqlContext.getConf("spark.sql.shuffle.partitions")
        partitionNum = df.rdd.getNumPartitions()
        sqlContext.setConf("spark.sql.shuffle.partitions", str(partitionNum))

        result_df = group_by_spark_df(df, columns, agg, join)

        agg_shards = spark_df_to_pd_sparkxshards(result_df)
        sqlContext.setConf("spark.sql.shuffle.partitions", defaultPartitionNum)
        return agg_shards

    def _to_spark_df_without_arrow(self):
        def f(iter):
            from bigdl.dllib.utils.log4Error import invalidInputError
            pdf_list = list(iter)
            invalidInputError(len(pdf_list) == 1,
                              f"For XShards of pandas dataframe, expects there is only 1"
                              f" pandas dataframe for each partition, but got {len(pdf_list)}")
            for pdf in pdf_list:
                np_records = pdf.to_records(index=False)
                return [r.tolist() for r in np_records]

        rdd = self.rdd.mapPartitions(f)
        column = self.get_schema()['columns']
        df = rdd.toDF(list(column))
        return df

    # to_spark_df adapted from pyspark
    # https://github.com/apache/spark/blob/master/python/pyspark/sql/pandas/conversion.py
    def to_spark_df(self) -> "SparkDataFrame":
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support to_spark_df on XShards of Pandas DataFrame")

        try:
            import pyarrow as pa
            sdf_schema = self._get_spark_df_schema()

            sqlContext = get_spark_sql_context(get_spark_context())
            timezone = sqlContext.getConf("spark.sql.session.timeZone")

            def f(iter):
                from bigdl.dllib.utils.log4Error import invalidInputError
                pdf_list = list(iter)
                invalidInputError(len(pdf_list) == 1,
                                  f"For XShards of pandas dataframe, expects there is only 1"
                                  f" pandas dataframe for each partition, but got {len(pdf_list)}")
                for pdf in pdf_list:
                    import os
                    import uuid
                    from pyspark.sql.pandas.types import to_arrow_type
                    from pyspark.sql.pandas.serializers import ArrowStreamPandasSerializer
                    from tempfile import NamedTemporaryFile

                    tmpFile = "/tmp/" + str(uuid.uuid1())
                    os.mkdir(tmpFile)

                    arrow_types = [to_arrow_type(f.dataType) for f in sdf_schema.fields]
                    arrow_data = [[(c, t) for (_, c), t in zip(pdf.items(), arrow_types)]]
                    col_by_name = True
                    safecheck = False
                    ser = ArrowStreamPandasSerializer(timezone, safecheck, col_by_name)

                    tempFile = NamedTemporaryFile(delete=False, dir=tmpFile)
                    try:
                        ser.dump_stream(arrow_data, tempFile)
                    finally:
                        tempFile.close()
                    return [tempFile.name]

            jiter = self.rdd.mapPartitions(f)
            from bigdl.dllib.utils.file_utils import callZooFunc

            df = callZooFunc("float", "orcaToDataFrame", jiter, sdf_schema.json(), sqlContext)
            return df
        except Exception as e:
            print(f"createDataFrame from shards attempted Arrow optimization failed as: {str(e)},"
                  f"Will try without Arrow optimization")
            return self._to_spark_df_without_arrow()

    def __len__(self) -> int:
        return self.rdd.map(lambda data: len(data) if hasattr(data, '__len__') else 1) \
            .reduce(lambda l1, l2: l1 + l2)

    def save_pickle(self, path: str, batchSize: int = 10) -> "SparkXShards":
        """

        Save this SparkXShards as a SequenceFile of serialized objects.
        The serializer used is pyspark.serializers.PickleSerializer, default batch size is 10.

        :param path: target path.
        :param batchSize: batch size for each sequence file chunk.
        """
        self.rdd.saveAsPickleFile(path, batchSize)
        return self

    def __del__(self):
        self._uncache()

    def __getitem__(self, key: str) -> "SparkXShards":
        def get_data(data):
            invalidInputError(hasattr(data, '__getitem__'),
                              "No selection operation available for this XShards")
            try:
                value = data[key]
            except:
                invalidInputError(False,
                                  "Invalid key for this XShards")
            return value

        return SparkXShards(self.rdd.map(get_data), transient=True)

    def _for_each(self, func: Callable, *args, **kwargs) -> "RDD[Any]":
        def utility_func(x, func, *args, **kwargs):
            try:
                result = func(x, *args, **kwargs)
            except Exception as e:
                return e
            return result

        result_rdd = self.rdd.map(lambda x: utility_func(x, func, *args, **kwargs))
        return result_rdd

    def get_schema(self) -> Optional[str]:
        if 'schema' in self.type:
            return self.type['schema']

        if 'class_name' not in self.type \
                or self.type['class_name'] == 'pandas.core.frame.DataFrame':
            class_name, pdf_schema, sdf_schema = self._get_schema_class_name()
            self.type['class_name'] = class_name
            self.type['schema'] = pdf_schema
            self.type['spark_df_schema'] = sdf_schema
            return self.type['schema']
        return None

    def _get_spark_df_schema(self):
        if 'spark_df_schema' in self.type:
            return self.type['spark_df_schema']

        if 'class_name' not in self.type \
                or self.type['class_name'] == 'pandas.core.frame.DataFrame':
            class_name, pdf_schema, sdf_schema = self._get_schema_class_name()
            self.type['class_name'] = class_name
            self.type['schema'] = pdf_schema
            self.type['spark_df_schema'] = sdf_schema
            return self.type['spark_df_schema']
        return None

    def _get_class_name(self):
        if 'class_name' in self.type:
            return self.type['class_name']
        else:
            class_name, schema, sdf_schema = self._get_schema_class_name()
            self.type['class_name'] = class_name
            self.type['schema'] = schema
            self.type['spark_df_schema'] = sdf_schema
            return self.type['class_name']

    def _set_class_name(self, class_name):
        if class_name and isinstance(class_name, str):
            self.type['class_name'] = class_name

    def _get_schema_class_name(self):
        class_name = self.type['class_name'] if 'class_name' in self.type else None
        import pyspark
        spark_version = pyspark.version.__version__
        major_version = spark_version.split(".")[0]

        def func(pdf):
            pdf_schema = None
            spark_df_schema = None
            _class_name = class_name
            if not _class_name:
                _class_name = pdf.__class__.__module__ + '.' + pdf.__class__.__name__

            if _class_name == 'pandas.core.frame.DataFrame':
                schema = [str(x) if not isinstance(x, str) else x for x in pdf.columns]
                pdf_schema = {'columns': schema, 'dtypes': pdf.dtypes}

                if major_version >= '3':
                    from pyspark.sql.pandas.types import from_arrow_type
                    from pyspark.sql.types import StructType

                    if isinstance(schema, (list, tuple)):
                        import pyarrow as pa
                        arrow_schema = pa.Schema.from_pandas(pdf, preserve_index=False)
                        struct = StructType()
                        for name, field in zip(schema, arrow_schema):
                            struct.add(
                                name, from_arrow_type(field.type), nullable=field.nullable
                            )
                        spark_df_schema = struct

            return (_class_name, pdf_schema, spark_df_schema)

        return self.rdd.map(lambda x: func(x)).first()

    def merge(self,
              right: "SparkXShards",
              how: str = "inner",
              on: Optional[str] = None) -> "SparkXShards":
        """
        Merge two SparkXShards into a single SparkXShards with a database-style join.

        :param right: The other SparkXShards to be merged.
        :param how: Type of merge. 'left', 'right', 'outer' or 'inner'. Default is 'inner'.
        :param on: Column name(s) to join on.
        :return: A new merged SparkXShards.
        """
        from bigdl.orca.data.utils import spark_df_to_pd_sparkxshards
        invalidInputError(isinstance(right, SparkXShards), "right should be a SparkXShards")

        left_df = self.to_spark_df()
        right_df = right.to_spark_df()
        merged = left_df.join(right_df, on=on, how=how)

        # count non-empty partitions
        nonEmptyPart = get_spark_context().accumulator(0)

        def f(iterator):
            isEmpty = 1
            for x in iterator:
                isEmpty = 0
                break
            nonEmptyPart.add(isEmpty == 0)
        merged.rdd.foreachPartition(f)

        # repartition evenly according to the index
        if nonEmptyPart.value != merged.rdd.getNumPartitions():
            merged_withIndex_rdd = merged.rdd.zipWithIndex().map(lambda p: (p[1], p[0]))
            merged = merged_withIndex_rdd.partitionBy(nonEmptyPart.value) \
                .map(lambda p: p[1]).toDF(merged.schema)
        mergedXShards = spark_df_to_pd_sparkxshards(merged)
        return mergedXShards

    def sample(self,
               frac: float,
               replace: bool=False,
               weights=None,
               random_state=None) -> "SparkXShards":
        """
        Samples from each pandas dataframe in old SparkXShards, Return a new SparkXShards  .

        :param frac: float,  Fraction of items to return.
        :param replace: bool, default False,
            Allow or disallow sampling of the same row more than once.
        :param weights: str or ndarray-like, optional
            Default 'None' results in equal probability weighting.
        :param random_state: int, array-like, BitGenerator, np.random.RandomState, optional
            If int, array-like, or BitGenerator (NumPy>=1.17), seed for
            random number generator
            If np.random.RandomState, use as numpy RandomState object.
        :return: a new SparkXShards.
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support sample() on"
                              " SparkXShards of Pandas DataFrame")

        def inner_sample(iter, frac, replace=False, weights=None, random_state=None):
            for df in iter:
                yield df.sample(
                    frac=frac, replace=replace, weights=weights, random_state=random_state)

        rdd1 = self.rdd.mapPartitions(lambda iter:
                                      inner_sample(iter, frac, replace, weights, random_state))
        return SparkXShards(rdd1)

    def select(self, cols: Union[str, List[str]]) -> "SparkXShards":
        """
        Select specific columns of each pandas dataframe in SparkXShards and
        return a new SparkXShards.

        :param cols: string or list string.
        :return: a new SparkXShards.
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support select() on"
                              " SparkXShards of Pandas DataFrame")

        if isinstance(cols, str):
            cols = [cols]
        invalidInputError(isinstance(cols, list), "cols should be str or list")

        columns = [c for c in self.rdd.first().columns]
        for c in cols:
            check_cols_exists(columns, c, "cols")

        return SparkXShards(self.rdd.map(lambda df: df[cols]))

    def describe(self, cols: Optional[Union[str, List[str]]]=None) -> "PandasDataFrame":
        """
        Computes basic statistics for numeric and string columns.

        This include count, mean, stddev, min, and max. If no columns are
        given, this function computes statistics for all numerical or string columns.

        :param cols: string or list string.
        :return: a panda dataframe of description.
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support select() on"
                              " SparkXShards of Pandas DataFrame")

        columns = [c for c in self.rdd.first().columns]
        cols = cols if cols else columns

        if isinstance(cols, str):
            cols = [cols]
        invalidInputError(isinstance(cols, list), "cols should be str or list")

        for c in cols:
            check_cols_exists(columns, c, "cols")

        spark_df = self.to_spark_df()

        description = spark_df.describe(*cols).toPandas()
        return description

    def head(self, n: int=5) -> 'PandasDataFrame':
        """
        Retrun first rows of the first element of a SparkXShards.

        :param n: int, default 5
        :return: same type as self.type['class_name']
            The first `n` rows of the first element of this SparkXShards.
        """

        return self.rdd.first().head(n)

    def concat_to_pdf(self, axis: int=0) -> "PandasDataFrame":
        """
        Concatenate all pandas dataframes in SparsXShards into one single pandas dataframe

        :param axis, integer, default 0
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support concat_to_pdf() on"
                              " XShards of Pandas DataFrame")

        dfs = self.rdd.collect()
        import pandas as pd
        return pd.concat(dfs, axis=axis)

    def sample_to_pdf(self,
                      frac: float,
                      replace: bool=False,
                      weights=None,
                      random_state=None,
                      axis: int=0) -> "PandasDataFrame":
        """
        Samples from each pandas dataframe in old SparkXShards, then concatenate into one single
        pandas dataframe and return it

        :param frac: float,  Fraction of items to return.
        :param replace: bool, default False,
            Allow or disallow sampling of the same row more than once.
        :param weights : str or ndarray-like, optional
            Default 'None' results in equal probability weighting.
        :param random_state : int, array-like, BitGenerator, np.random.RandomState, optional
            If int, array-like, or BitGenerator (NumPy>=1.17), seed for
            random number generator
            If np.random.RandomState, use as numpy RandomState object.
        :param axis, integer, default 0

        :return: a pandas dataframe.
        """
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support sample_to_pdf() on"
                              " XShards of Pandas DataFrame")
        sampled = self.sample(
            frac=frac, replace=replace, weights=weights, random_state=random_state)
        pdf = sampled.concat_to_pdf(axis=axis)
        return pdf

    def stack_feature_labels(self) -> "SparkXShards":
        """
        Stack tuple of features and labels in each partition into an ndarray for
        Orca Estimator traning

        :return: SparkXShards.
        """
        if self._get_class_name() != "builtins.tuple":
            invalidInputError(False,
                              "Currently only support stack_feature_labels() on"
                              " XShards of tuple of features and labels")

        def per_partition(iterator):
            features = []
            labels = []
            for it in iterator:
                feature, label = it[0], it[1]
                features.append(feature)
                labels.append(label)
            out = {'x': np.array(features).astype(np.float32),
                   'y': np.array(labels).astype(np.float32)}
            return [out]
        rdd = self.rdd.mapPartitions(lambda x: per_partition(x))
        return SparkXShards(rdd)


class SharedValue(object):
    def __init__(self, data) -> None:
        sc = OrcaContext.get_spark_context()
        self.broadcast_data = sc.broadcast(data)
        self._value = None

    @property
    def value(self):
        self._value = self.broadcast_data.value
        return self._value

    def unpersist(self):
        self.broadcast_data.unpersist()


def spark_df_to_ray_dataset(df: "SparkDataFrame") -> "Dataset":
    """
    Convert a Spark DataFrame to Ray Dataset. The block number of ray datasets equals to the
    partition number of the input DataFrame.
    :param df: A Spark dataframe.
    :return: A Ray Dataset holding Arrow records read from the dataframe.
    """
    spark_xshards = spark_df_to_pd_sparkxshards(df)
    ray_dataset = spark_xshards_to_ray_dataset(spark_xshards)
    return ray_dataset

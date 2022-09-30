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
from bigdl.dllib.nncontext import init_nncontext, ZooContext
from bigdl.dllib.utils.common import *
from bigdl.dllib.utils import nest
from bigdl.dllib.utils.log4Error import *

import numpy as np


class XShards(object):
    """

    A collection of data which can be pre-processed in parallel.
    """
    def transform_shard(self, func, *args):
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
    def load_pickle(cls, path, minPartitions=None):
        """

        Load XShards from pickle files.

        :param path: The pickle file path/directory
        :param minPartitions: The minimum partitions for the XShards
        :return: SparkXShards object
        """
        sc = init_nncontext()
        return SparkXShards(sc.pickleFile(path, minPartitions))

    @staticmethod
    def partition(data, num_shards=None):
        """

        Partition local in memory data and form a SparkXShards

        :param data: np.ndarray, a tuple, list, dict of np.ndarray, or a nested structure
        made of tuple, list, dict with ndarray as the leaf value
        :param num_shards: the number of shards that the data will be partitioned into
        :return: a SparkXShards
        """
        sc = init_nncontext()
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
            data_to_be_shard = []
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

    A collection of data which can be pre-processed in parallel on Spark
    """
    def __init__(self, rdd, transient=False, class_name=None):
        self.rdd = rdd
        self.user_cached = False
        if transient:
            self.eager = False
        else:
            self.eager = OrcaContext._eager_mode
            self.rdd.cache()
        if self.eager:
            self.compute()
        self.type = {}
        if class_name:
            self.type['class_name'] = class_name

    def transform_shard(self, func, *args):
        """

        Return a new SparkXShards by applying a function to each shard of this SparkXShards

        :param func: python function to process data. The first argument is the data shard.
        :param args: other arguments in this function.
        :return: a new SparkXShards.
        """
        def transform(iter, func, *args):
            for x in iter:
                yield func(x, *args)

        transformed_shard = SparkXShards(self.rdd.mapPartitions(lambda iter:
                                                                transform(iter, func, *args)))
        self._uncache()
        return transformed_shard

    def collect(self):
        """

        Returns a list that contains all of the elements in this SparkXShards

        :return: a list of data elements.
        """
        return self.rdd.collect()

    def cache(self):
        """

        Persist this SparkXShards in memory

        :return:
        """
        self.user_cached = True
        self.rdd.cache()
        return self

    def uncache(self):
        """

        Make this SparkXShards as non-persistent, and remove all blocks for it from memory

        :return:
        """
        self.user_cached = False
        if self.is_cached():
            try:
                self.rdd.unpersist()
            except Py4JError:
                print("Try to unpersist an uncached rdd")
        return self

    def _uncache(self):
        if not self.user_cached:
            self.uncache()

    def is_cached(self):
        return self.rdd.is_cached

    def compute(self):
        self.rdd.count()
        return self

    def num_partitions(self):
        """

        Get number of partitions for this SparkXShards.

        :return: number of partitions.
        """
        return self.rdd.getNumPartitions()

    def repartition(self, num_partitions):
        """

        Return a new SparkXShards that has exactly num_partitions partitions.

        :param num_partitions: target number of partitions
        :return: a new SparkXShards object.
        """
        if self._get_class_name() == 'pandas.core.frame.DataFrame':
            import pandas as pd

            if num_partitions > self.rdd.getNumPartitions():
                rdd = self.rdd\
                    .flatMap(lambda df: df.apply(lambda row: (row[0], row.values.tolist()), axis=1)
                             .values.tolist())\
                    .partitionBy(num_partitions)

                schema = self.get_schema()

                def merge_rows(iter):
                    data = [value[1] for value in list(iter)]
                    if data:
                        df = pd.DataFrame(data=data, columns=schema['columns'])\
                            .astype(schema['dtypes'])
                        return [df]
                    else:
                        # no data in this partition
                        return iter
                repartitioned_shard = SparkXShards(rdd.mapPartitions(merge_rows))
            else:
                def combine_df(iter):
                    dfs = list(iter)
                    if len(dfs) > 0:
                        return [pd.concat(dfs)]
                    else:
                        return iter
                rdd = self.rdd.coalesce(num_partitions)
                repartitioned_shard = SparkXShards(rdd.mapPartitions(combine_df))
        elif self._get_class_name() == 'builtins.list':
            if num_partitions > self.rdd.getNumPartitions():
                rdd = self.rdd \
                    .flatMap(lambda data: data) \
                    .repartition(num_partitions)

                repartitioned_shard = SparkXShards(rdd.mapPartitions(
                    lambda iter: [list(iter)]))
            else:
                rdd = self.rdd.coalesce(num_partitions)
                from functools import reduce
                repartitioned_shard = SparkXShards(rdd.mapPartitions(
                    lambda iter: [reduce(lambda l1, l2: l1 + l2, iter)]))
        elif self._get_class_name() == 'numpy.ndarray':
            elem = self.rdd.first()
            shape = elem.shape
            dtype = elem.dtype
            if len(shape) > 0:
                if num_partitions > self.rdd.getNumPartitions():
                    rdd = self.rdd\
                        .flatMap(lambda data: list(data))\
                        .repartition(num_partitions)

                    repartitioned_shard = SparkXShards(rdd.mapPartitions(
                        lambda iter: np.stack([list(iter)], axis=0)
                        .astype(dtype)))
                else:
                    rdd = self.rdd.coalesce(num_partitions)
                    from functools import reduce
                    repartitioned_shard = SparkXShards(rdd.mapPartitions(
                        lambda iter: [np.concatenate(list(iter), axis=0)]))
            else:
                repartitioned_shard = SparkXShards(self.rdd.repartition(num_partitions))
        elif self._get_class_name() == "builtins.dict":
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
                if num_partitions > self.rdd.getNumPartitions():
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
                    rdd = self.rdd.flatMap(lambda data: dict_to_unbatched_list(data))\
                        .repartition(num_partitions)
                    repartitioned_shard = SparkXShards(rdd.mapPartitions(
                        lambda iter: to_batched_dict(iter)))
                else:
                    rdd = self.rdd.coalesce(num_partitions)

                    def merge_list_of_dict(iter):
                        iter_list = list(iter)
                        return [{k: np.concatenate([d[k] for d in iter_list], axis=0)
                                 for k in keys}]
                    repartitioned_shard = SparkXShards(rdd.mapPartitions(
                        lambda iter: merge_list_of_dict(iter)))
            else:
                repartitioned_shard = SparkXShards(self.rdd.repartition(num_partitions))
        else:
            repartitioned_shard = SparkXShards(self.rdd.repartition(num_partitions))
        self._uncache()
        return repartitioned_shard

    def partition_by(self, cols, num_partitions=None):
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
                if cols not in schema['columns']:
                    invalidInputError(False,
                                      "The partition column is not in the DataFrame")
                # change data to key value pairs
                rdd = self.rdd.flatMap(
                    lambda df: df.apply(lambda row: (row[cols], row.values.tolist()), axis=1)
                    .values.tolist())

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

    def unique(self):
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
            # we may support numpy or other types later
            invalidInputError(False,
                              "Currently only support unique() on XShards of Pandas Series")

    def deduplicates(self):
        if self._get_class_name() == 'pandas.core.frame.DataFrame':
            import pandas as pd
            df = self.to_spark_df()
            distinctDF = df.distinct()
            data_shards = spark_df_to_pd_sparkxshards(distinctDF)
            return data_shards
        else:
            # we may support numpy or other types later
            invalidInputError(False,
                              "Currently only support dedup() on XShards of Pandas DataFrame")

    def assembleFeatureLabelCols(self, featureCols, labelCols):
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

    def split(self):
        """

        Split SparkXShards into multiple SparkXShards.
        Each element in the SparkXShards needs be a list or tuple with same length.

        :return: Splits of SparkXShards. If element in the input SparkDataShard is not
                list or tuple, return list of input SparkDataShards.
        """
        # get number of splits
        list_split_length = self.rdd.map(lambda data: len(data) if isinstance(data, list) or
                                         isinstance(data, tuple) else 1).collect()
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

    def zip(self, other):
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
    def to_spark_df(self):
        if self._get_class_name() != 'pandas.core.frame.DataFrame':
            invalidInputError(False,
                              "Currently only support to_spark_df on XShards of Pandas DataFrame")

        try:
            import pyarrow as pa
            sdf_schema = self._get_spark_df_schema()

            sqlContext = get_spark_sql_context(get_spark_context())
            timezone = sqlContext._conf.sessionLocalTimeZone()

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
                    arrow_data = [[(c, t) for (_, c), t in zip(pdf.iteritems(), arrow_types)]]
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

    def __len__(self):
        return self.rdd.map(lambda data: len(data) if hasattr(data, '__len__') else 1)\
            .reduce(lambda l1, l2: l1 + l2)

    def save_pickle(self, path, batchSize=10):
        """

        Save this SparkXShards as a SequenceFile of serialized objects.
        The serializer used is pyspark.serializers.PickleSerializer, default batch size is 10.

        :param path: target path.
        :param batchSize: batch size for each sequence file chunk.
        """
        self.rdd.saveAsPickleFile(path, batchSize)
        return self

    def __del__(self):
        self.uncache()

    def __getitem__(self, key):
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

    def _for_each(self, func, *args, **kwargs):
        def utility_func(x, func, *args, **kwargs):
            try:
                result = func(x, *args, **kwargs)
            except Exception as e:
                return e
            return result
        result_rdd = self.rdd.map(lambda x: utility_func(x, func, *args, **kwargs))
        return result_rdd

    def get_schema(self):
        if 'schema' in self.type:
            return self.type['schema']

        if 'class_name' not in self.type\
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

        if 'class_name' not in self.type\
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


class SharedValue(object):
    def __init__(self, data):
        sc = init_nncontext()
        self.broadcast_data = sc.broadcast(data)
        self._value = None

    @property
    def value(self):
        self._value = self.broadcast_data.value
        return self._value

    def unpersist(self):
        self.broadcast_data.unpersist()


def spark_df_to_ray_dataset(df):
    """
    Convert a Spark DataFrame to Ray Dataset. The block number of ray datasets equals to the
    partition number of the input DataFrame.
    :param df: A Spark dataframe.
    :return: A Ray Dataset holding Arrow records read from the dataframe.
    """
    spark_xshards = spark_df_to_pd_sparkxshards(df)
    ray_dataset = spark_xshards_to_ray_dataset(spark_xshards)
    return ray_dataset

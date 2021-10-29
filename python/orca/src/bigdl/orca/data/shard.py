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
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils import nest


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
                raise ValueError("The length of data {} is smaller than the total number "
                                 "of shards {}. Please adjust the num_shards option to be "
                                 "at most {}.".format(data.shape[0], shard_num, data.shape[0]))
            arrays = np.array_split(data, shard_num)
            rdd = sc.parallelize(arrays)
        else:
            assert type(data) in supported_types, type_err_msg
            flattened = nest.flatten(data)
            data_length = len(flattened[0])
            data_to_be_shard = []
            if data_length < shard_num:
                raise ValueError("The length of data {} is smaller than the total number "
                                 "of shards {}. Please adjust the num_shards option to be "
                                 "at most {}.".format(data_length, shard_num, data_length))
            for i in range(shard_num):
                data_to_be_shard.append([])
            for x in flattened:
                assert len(x) == data_length, \
                    "the ndarrays in data must all have the same size in first dimension, " \
                    "got first ndarray of size {} and another {}".format(data_length, len(x))
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
    def __init__(self, rdd, transient=False):
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

                schema = self._get_schema()

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
        elif self._get_class_name() == 'list':
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
            schema = self._get_schema()
            # if partition by a column
            if isinstance(cols, str):
                if cols not in schema['columns']:
                    raise Exception("The partition column is not in the DataFrame")
                # change data to key value pairs
                rdd = self.rdd.flatMap(
                    lambda df: df.apply(lambda row: (row[cols], row.values.tolist()), axis=1)
                    .values.tolist())

                partition_num = self.rdd.getNumPartitions() if not num_partitions \
                    else num_partitions
                # partition with key
                partitioned_rdd = rdd.partitionBy(partition_num)
            else:
                raise Exception("Only support partition by a column name")

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
            raise Exception("Currently only support partition by for XShards"
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
            raise Exception("Currently only support unique() on XShards of Pandas Series")

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
            raise Exception("Cannot split this XShards because its partitions "
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
        assert isinstance(other, SparkXShards), "other should be a SparkXShards"
        assert self.num_partitions() == other.num_partitions(), \
            "The two SparkXShards should have the same number of partitions"
        try:
            rdd = self.rdd.zip(other.rdd)
            zipped_shard = SparkXShards(rdd)
            other._uncache()
            self._uncache()
            return zipped_shard
        except Exception:
            raise ValueError("The two SparkXShards should have the same number of elements "
                             "in each partition")

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
            assert hasattr(data, '__getitem__'), \
                "No selection operation available for this XShards"
            try:
                value = data[key]
            except:
                raise Exception("Invalid key for this XShards")
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

    def _get_schema(self):
        if 'schema' in self.type:
            return self.type['schema']
        else:
            if self._get_class_name() == 'pandas.core.frame.DataFrame':
                import pandas as pd
                columns, dtypes = self.rdd.map(lambda x: (x.columns, x.dtypes)).first()
                self.type['schema'] = {'columns': columns, 'dtypes': dtypes}
                return self.type['schema']
            return None

    def _get_class_name(self):
        if 'class_name' in self.type:
            return self.type['class_name']
        else:
            self.type['class_name'] = self._for_each(get_class_name).first()
            return self.type['class_name']


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

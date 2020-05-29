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

from zoo.orca.data.utils import *


class XShards(object):
    """
    A collection of data which can be pre-processed parallelly.
    """

    def transform_shard(self, func, *args):
        """
        Transform each shard in the XShards using func
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


class RayXShards(XShards):
    """
    A collection of data which can be pre-processed parallelly on Ray
    """
    def __init__(self, partitions):
        self.partitions = partitions
        self.shard_list = flatten([partition.shard_list for partition in partitions])

    def transform_shard(self, func, *args):
        """
        Transform each shard in the XShards using func
        :param func: pre-processing function.
        In the function, the element object should be the first argument
        :param args: rest arguments for the pre-processing function
        :return: this DataShard
        """
        import ray
        done_ids, undone_ids = ray.wait([shard.transform.remote(func, *args)
                                         for shard in self.shard_list],
                                        num_returns=len(self.shard_list))
        assert len(undone_ids) == 0
        return self

    def collect(self):
        """
        Returns a list that contains all of the elements in this XShards
        :return: list of elements
        """
        import ray
        return ray.get([shard.get_data.remote() for shard in self.shard_list])

    def num_partitions(self):
        return len(self.partitions)

    def repartition(self, num_partitions):
        """
        Repartition XShards.
        :param num_partitions: number of partitions
        :return: this XShards
        """
        shards_partitions = list(chunk(self.shard_list, num_partitions))
        self.partitions = [RayPartition(shards) for shards in shards_partitions]
        return self

    def get_partitions(self):
        """
        Return partition list of the XShards
        :return: partition list
        """
        return self.partitions


class RayPartition(object):
    """
    Partition of RayXShards
    """

    def __init__(self, shard_list):
        self.shard_list = shard_list

    def get_data(self):
        return [shard.get_data.remote() for shard in self.shard_list]


class SparkXShards(XShards):
    def __init__(self, rdd):
        self.rdd = rdd

    def transform_shard(self, func, *args):
        return SparkXShards(self.rdd.map(lambda data: func(data, *args)))

    def collect(self):
        return self.rdd.collect()

    def num_partitions(self):
        return self.rdd.getNumPartitions()

    def repartition(self, num_partitions):
        return SparkXShards(self.rdd.repartition(num_partitions))

    def partition_by(self, cols, num_partitions=None):
        import pandas as pd
        elem_class, columns = self.rdd.map(
            lambda data: (type(data), data.columns) if isinstance(data, pd.DataFrame)
            else (type(data), None)).first()
        if issubclass(elem_class, pd.DataFrame):
            # if partition by a column
            if isinstance(cols, str):
                if cols not in columns:
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
                    df = pd.DataFrame(data=data, columns=columns)
                    return [df]
                else:
                    # no data in this partition
                    return []
            # merge records to df in each partition
            return SparkXShards(partitioned_rdd.mapPartitions(merge))
        else:
            raise Exception("Currently only support partition by for XShards"
                            " of Pandas DataFrame")

    def unique(self, key):
        import pandas as pd
        elem_class, columns = self.rdd.map(
            lambda data: (type(data), data.columns) if isinstance(data, pd.DataFrame)
            else (type(data), None)).first()
        if issubclass(elem_class, pd.DataFrame):
            if key is None:
                raise Exception("Cannot apply unique operation on Datashards of Pandas Dataframe"
                                " without column name")
            if key in columns:
                rdd = self.rdd.map(lambda df: df[key].unique())
                import numpy as np
                result = rdd.reduce(lambda list1, list2: pd.unique(np.concatenate((list1, list2),
                                                                                  axis=0)))
                return result
            else:
                raise Exception("The select key is not in the DataFrame in this Datashards")
        else:
            # we may support numpy or other types later
            raise Exception("Currently only support unique() on Datashards of Pandas DataFrame")

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
                return [SparkXShards(self.rdd.map(get_data(i)))
                        for i in range(list_split_length[0])]
            else:
                return [self]

    def save_pickle(self, path, batchSize=10):
        self.rdd.saveAsPickleFile(path, batchSize)
        return self

    @classmethod
    def load_pickle(cls, path, sc, minPartitions=None):
        return SparkXShards(sc.pickleFile(path, minPartitions))

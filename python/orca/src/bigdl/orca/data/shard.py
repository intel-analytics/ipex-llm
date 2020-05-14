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


class DataShards(object):
    """
    A collection of data which can be pre-processed parallelly.
    """

    def transform_shard(self, func, *args):
        """
        Transform each shard in the DataShards using func
        :param func: pre-processing function
        :param args: arguments for the pre-processing function
        :return: DataShard
        """
        pass

    def collect(self):
        """
        Returns a list that contains all of the elements in this DataShards
        :return: list of elements
        """
        pass


class RayDataShards(DataShards):
    """
    A collection of data which can be pre-processed parallelly on Ray
    """
    def __init__(self, partitions):
        self.partitions = partitions
        self.shard_list = flatten([partition.shard_list for partition in partitions])

    def transform_shard(self, func, *args):
        """
        Transform each shard in the DataShards using func
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
        Returns a list that contains all of the elements in this DataShards
        :return: list of elements
        """
        import ray
        return ray.get([shard.get_data.remote() for shard in self.shard_list])

    def repartition(self, num_partitions):
        """
        Repartition DataShards.
        :param num_partitions: number of partitions
        :return: this DataShards
        """
        shards_partitions = list(chunk(self.shard_list, num_partitions))
        self.partitions = [RayPartition(shards) for shards in shards_partitions]
        return self

    def get_partitions(self):
        """
        Return partition list of the DataShards
        :return: partition list
        """
        return self.partitions


class RayPartition(object):
    """
    Partition of RayDataShards
    """

    def __init__(self, shard_list):
        self.shard_list = shard_list

    def get_data(self):
        return [shard.get_data.remote() for shard in self.shard_list]


class SparkDataShards(DataShards):
    def __init__(self, rdd):
        self.rdd = rdd

    def transform_shard(self, func, *args):
        self.rdd = self.rdd.map(func(*args))
        return self

    def collect(self):
        return self.rdd.collect()

    def repartition(self, num_partitions):
        self.rdd = self.rdd.repartition(num_partitions)
        return self

    def partition_by(self, cols, num_partitions=None):
        import pandas as pd
        class_name, columns = self.rdd.map(
            lambda data: (get_class_name(data), data.columns) if isinstance(data, pd.DataFrame)
            else (get_class_name(data), None)).first()
        if class_name == 'pandas.core.frame.DataFrame':
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
            self.rdd = partitioned_rdd.mapPartitions(merge)
            return self
        else:
            raise Exception("Currently only support partition by for Datashards"
                            " of Pandas DataFrame")

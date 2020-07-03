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
from py4j.protocol import Py4JError

from zoo.orca.data.utils import *
from zoo.common.nncontext import init_nncontext
from zoo import ZooContext


class XShards(object):
    """
    A collection of data which can be pre-processed in parallel.
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
    A collection of data which can be pre-processed in parallel on Ray
    """
    def __init__(self, partitions):
        assert all([isinstance(partition, RayPartition) for partition in partitions]), \
            "partitions should be list of RayPartition"
        self.partitions = partitions

    def transform_shard(self, func, *args):
        raise Exception("Transform is not supported for RayXShards")

    def collect(self):
        return [partition.get_data() for partition in self.partitions]

    def num_partitions(self):
        return len(self.partitions)

    def get_partitions(self):
        """
        Return the list of RayPartition of the RayXShards
        """
        return self.partitions

    def colocate_actors(self, actors):
        """
        Sort Ray actors and RayPartitions by node_ip so that each actor is colocated
        with the data partition on the same node.
        """
        if self.partitions[0].node_ip:
            # Assume that the partitions are already sorted by node_ip
            import ray
            actor_ips = ray.get([actor.get_node_ip.remote() for actor in actors])
            actor_zip_ips = list(zip(actors, actor_ips))
            actor_zip_ips.sort(key=lambda x: x[1])
            for i in range(len(actors)):
                actor_ip = actor_zip_ips[i][1]
                partition_ip = self.partitions[i].node_ip
                assert actor_ip == partition_ip
            return [actor_ip[0] for actor_ip in actor_zip_ips]
        else:
            return actors


class RayPartition(object):
    """
    A partition of RayXShards containing the plasma ObjectID, the plasma object_store_address,
    and the node of the partition.
    """
    def __init__(self, object_id, node_ip, object_store_address):
        # The object_id would contain a list of data from a partition of Spark RDD.
        self.object_id = object_id
        self.node_ip = node_ip
        self.object_store_address = object_store_address

    def get_data(self):
        import pyarrow.plasma as plasma
        # Default num_retries=-1 would try 80 times.
        self.client = plasma.connect(self.object_store_address, num_retries=5)
        return self.client.get(self.object_id)

    def __del__(self):
        if self.object_store_address:
            import logging
            logging.basicConfig(level=logging.WARNING)
            logger = logging.getLogger()
            try:
                if "client" not in self.__dict__:
                    import pyarrow.plasma as plasma
                    self.client = plasma.connect(self.object_store_address, num_retries=5)
                if self.client.contains(self.object_id):
                    self.client.delete([self.object_id])
                assert not self.client.contains(self.object_id)
                logger.info("Removed data from plasma object store on node " + str(self.node_ip))
                self.client.disconnect()
                del self.client
            except Exception as e:
                logger.warning(e)
                logger.warning("Error occurred when removing the data from the plasma store "
                               "on node " + str(self.node_ip))


class SparkXShards(XShards):
    def __init__(self, rdd, transient=False):
        self.rdd = rdd
        self.user_cached = False
        if transient:
            self.eager = False
        else:
            self.eager = ZooContext._orca_eager_mode
            self.rdd.cache()
        if self.eager:
            self.compute()
        self.type = {}

    def transform_shard(self, func, *args):
        transformed_shard = SparkXShards(self.rdd.map(lambda data: func(data, *args)))
        self._uncache()
        return transformed_shard

    def collect(self):
        return self.rdd.collect()

    def cache(self):
        self.user_cached = True
        self.rdd.cache()
        return self

    def uncache(self):
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
        return self.rdd.getNumPartitions()

    def repartition(self, num_partitions):
        repartitioned_shard = SparkXShards(self.rdd.repartition(num_partitions))
        self._uncache()
        return repartitioned_shard

    def partition_by(self, cols, num_partitions=None):
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
                    df = pd.DataFrame(data=data, columns=schema['columns'])
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

    def __len__(self):
        return self.rdd.map(lambda data: len(data) if hasattr(data, '__len__') else 1)\
            .reduce(lambda l1, l2: l1 + l2)

    def save_pickle(self, path, batchSize=10):
        self.rdd.saveAsPickleFile(path, batchSize)
        return self

    @classmethod
    def load_pickle(cls, path, minPartitions=None):
        sc = init_nncontext()
        return SparkXShards(sc.pickleFile(path, minPartitions))

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

    # Tested on pyarrow 0.17.0; 0.16.0 would get errors.
    def to_ray(self):
        from zoo.ray import RayContext
        ray_ctx = RayContext.get()
        object_store_address = ray_ctx.address_info["object_store_address"]

        def put_to_plasma(ids):
            def f(index, iterator):
                import pyarrow.plasma as plasma
                from zoo.orca.data.utils import get_node_ip
                res = list(iterator)
                client = plasma.connect(object_store_address)
                target_id = ids[index]
                # If the ObjectID exists in plasma, we assume a task trial
                # succeeds and the data is already in the object store.
                if not client.contains(target_id):
                    object_id = client.put(res, target_id)
                    assert object_id == target_id, \
                        "Errors occurred when putting data into plasma object store"
                client.disconnect()
                yield target_id, get_node_ip()
            return f

        # Create plasma ObjectIDs beforehand instead of creating a random one every time to avoid
        # memory leak in case errors occur when putting data into plasma and Spark would retry.
        # ObjectIDs in plasma is a byte string of length 20 containing characters and numbers.
        # The random generation of ObjectIDs is often good enough to ensure unique IDs.
        import pyarrow.plasma as plasma
        object_ids = [plasma.ObjectID.from_random() for i in range(self.rdd.getNumPartitions())]
        object_id_node_ips = self.rdd.mapPartitionsWithIndex(put_to_plasma(object_ids)).collect()
        self.uncache()
        # Sort the data according to the node_ips.
        object_id_node_ips.sort(key=lambda x: x[1])
        partitions = [RayPartition(object_id=id_ip[0], node_ip=id_ip[1],
                                   object_store_address=object_store_address)
                      for id_ip in object_id_node_ips]
        return RayXShards(partitions)

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
                self.type['schema'] = {'columns': columns, 'dtype': dtypes}
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

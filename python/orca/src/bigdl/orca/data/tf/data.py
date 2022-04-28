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
import tensorflow as tf

from bigdl.orca.tfpark.tf_dataset import TensorMeta
from bigdl.dllib.utils import nest
from bigdl.orca.data import SparkXShards


class Dataset(object):

    """
    Represents a distributed set of elements backed by an RDD,
    which is created by applying tensorflow dataset transformations
    on each partitions.
    """

    def __init__(self, xshards, create_dataset_fn):
        self.xshards = xshards
        self.create_dataset_fn = create_dataset_fn

    def as_graph_rdd(self, batch_per_shard, drop_remainder=True):

        create_dataset_fn = self.create_dataset_fn

        def to_dataset(iter):
            data_list = list(iter)

            import tensorflow as tf
            if not data_list:
                return []

            datasets = [create_dataset_fn(data) for data in data_list]
            from functools import reduce
            dataset = reduce(lambda x, y: x.concatenate(y), datasets)
            dataset = dataset.batch(batch_per_shard, drop_remainder)
            iterator = dataset.make_initializable_iterator()
            train_next_ops = nest.flatten(iterator.get_next())
            output_types = [t for t in nest.flatten(dataset.output_types)]
            output_types_enum = [t.as_datatype_enum for t in output_types]

            init_op_name = iterator.initializer.name
            table_init_op = tf.tables_initializer().name
            output_names = [op.name for op in train_next_ops]

            graph = train_next_ops[0].graph

            flatten_shapes = nest.flatten(dataset.output_shapes)

            flatten_shapes = [shape[1:] for shape in flatten_shapes]

            flatten_tensor_structure = [TensorMeta(dtype=output_types[i],
                                                   shape=list(flatten_shapes[i]),
                                                   name="zoo_input_{}".format(i))
                                        for i in range(len(flatten_shapes))]
            structure = dataset.output_types
            if isinstance(structure, tf.DType):
                structure = (structure,)
            tensor_structure = nest.pack_sequence_as(structure,
                                                     flatten_tensor_structure)

            meta_info = {
                "init_op_name": init_op_name,
                "table_init_op": table_init_op,
                "output_names": output_names,
                "output_types": output_types_enum,
                "tensor_structure": tensor_structure
            }

            return [(bytearray(graph.as_graph_def().SerializeToString()), meta_info)]

        graph_rdd_and_meta = self.xshards.rdd.mapPartitions(to_dataset)
        return graph_rdd_and_meta

    def as_tf_dataset(self):
        create_dataset_fn = self.create_dataset_fn

        def to_dataset(iter):

            data_list = list(iter)
            if not data_list:
                return []

            from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph
            datasets = [create_dataset_fn(data) for data in data_list]
            from functools import reduce
            dataset = reduce(lambda x, y: x.concatenate(y), datasets)
            ds_def = serialize_dataset_to_graph(dataset).numpy()
            elem_spec = dataset.element_spec
            return [{"ds_def": ds_def, "elem_spec": elem_spec}]

        tf_dataset_rdd = self.xshards.rdd.mapPartitions(to_dataset)
        return tf_dataset_rdd

    @staticmethod
    def from_tensor_slices(xshards):
        return TensorSliceDataset(xshards)

    @staticmethod
    def from_tensor_slices_with_tbl(tbl):
        from bigdl.friesian.feature import FeatureTable
        from bigdl.friesian.feature.utils import featuretable_to_xshards
        assert isinstance(tbl, FeatureTable), "only Friesian FeatureTable is supported"
        xshards = featuretable_to_xshards(tbl)
        return TensorSliceDataset(xshards)

    def map(self, map_func):

        return MapDataset(self, map_func)


class TensorSliceDataset(Dataset):

    def __init__(self, xshards):
        assert isinstance(xshards, SparkXShards), \
            "only datasets backed by a SparkXShards are supported"

        self.xshards = xshards

        def create_dataset_fn(data):
            return tf.data.Dataset.from_tensor_slices(data)
        super().__init__(xshards, create_dataset_fn)


class MapDataset(Dataset):

    def __init__(self, input_dataset, map_func):

        create_pre_dataset_fn = input_dataset.create_dataset_fn

        def create_dataset_fn(data):
            dataset = create_pre_dataset_fn(data)
            return dataset.map(map_func)
        super().__init__(xshards=input_dataset.xshards,
                         create_dataset_fn=create_dataset_fn)

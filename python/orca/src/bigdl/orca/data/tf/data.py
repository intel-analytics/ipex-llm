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
import tensorflow as tf

from zoo.tfpark.tf_dataset import TensorMeta
from zoo.util import nest
from zoo import getOrCreateSparkContext, get_node_and_core_number
from zoo.common import callZooFunc
from zoo.feature.common import FeatureSet
from zoo.orca.data import SparkXShards
from zoo.tfpark import TFDataset


class TFDataDataset2(TFDataset):

    def __init__(self, dataset, batch_size,
                 batch_per_thread,
                 validation_dataset=None, intra_threads=None, inter_threads=None):

        node_num, core_num = get_node_and_core_number()

        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        if intra_threads is None:
            self.intra_threads = core_num

        if inter_threads is None:
            self.inter_threads = 1

        if batch_size > 0:
            num_parts = dataset.xshards.num_partitions()
            if num_parts != node_num:
                dataset.xshards = dataset.xshards.repartition(node_num)
            assert batch_size % node_num == 0, \
                "batch_size should be a multiple of num_shards, got" \
                " batch_size {}, node_num {}".format(batch_size, node_num)
            batch_per_shard = batch_size // node_num
            self.drop_remainder = True
        elif batch_per_thread > 0:
            batch_per_shard = batch_per_thread
            self.drop_remainder = False
        else:
            raise ValueError("one of batch_size or batch_per_thread must be larger than 0")

        self.rdd = dataset.as_graph_rdd(batch_per_shard,
                                        drop_remainder=self.drop_remainder).cache()
        meta_info = self.rdd.map(lambda x: x[1]).first()
        tensor_structure = meta_info["tensor_structure"]
        self.init_op_name = meta_info["init_op_name"]
        self.output_names = meta_info["output_names"]
        self.output_types = meta_info["output_types"]
        self.table_init_op = meta_info["table_init_op"]

        if validation_dataset is not None:
            self.val_rdd = validation_dataset.as_graph_rdd(batch_per_shard, False).cache()
            meta_info = self.val_rdd.map(lambda x: x[1]).first()
            self.val_init_op_name = meta_info["init_op_name"]
            self.val_output_names = meta_info["output_names"]
            self.val_output_types = meta_info["output_types"]
        else:
            self.val_rdd = None
            self.val_init_op_name = None
            self.val_output_names = None
            self.val_output_types = None

        super().__init__(tensor_structure, batch_size=batch_size,
                         batch_per_thread=batch_per_thread,
                         hard_code_batch_size=False)
        self.shard_index_op_name = None
        self.validation_dataset = validation_dataset

    def _get_prediction_data(self):
        assert not self.drop_remainder, \
            "sanity check: drop_remainder should be false in this case," \
            " otherwise please report a bug"
        jvalue = callZooFunc("float", "createMiniBatchRDDFromTFDataset",
                             self.rdd.map(lambda x: x[0]), self.init_op_name, self.table_init_op,
                             self.output_names, self.output_types, self.shard_index_op_name)
        rdd = jvalue.value().toJavaRDD()
        return rdd

    def _get_evaluation_data(self):
        jvalue = callZooFunc("float", "createMiniBatchRDDFromTFDatasetEval",
                             self.rdd.map(lambda x: x[0]), self.init_op_name, self.table_init_op,
                             self.output_names,
                             self.output_types, self.shard_index_op_name)
        rdd = jvalue.value().toJavaRDD()
        return rdd

    def _get_training_data(self):
        jvalue = callZooFunc("float", "createTFDataFeatureSet",
                             self.rdd.map(lambda x: x[0]), self.init_op_name, self.table_init_op,
                             self.output_names, self.output_types, self.shard_index_op_name,
                             self.inter_threads, self.intra_threads)
        return FeatureSet(jvalue=jvalue)

    def _get_validation_data(self):
        if self.validation_dataset is not None:
            jvalue = callZooFunc("float", "createTFDataFeatureSet",
                                 self.val_rdd.map(lambda x: x[0]), self.init_op_name,
                                 self.table_init_op, self.output_names,
                                 self.output_types, self.shard_index_op_name,
                                 self.inter_threads, self.intra_threads)
            return FeatureSet(jvalue=jvalue)
        return None

    def get_num_partitions(self):
        return self.rdd.getNumPartitions()


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

    @staticmethod
    def from_tensor_slices(xshards):
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

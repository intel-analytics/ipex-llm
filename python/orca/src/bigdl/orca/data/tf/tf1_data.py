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

from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.feature.common import FeatureSet
from bigdl.orca.tfpark import TFDataset
from bigdl.dllib.utils.log4Error import *


class TF1Dataset(TFDataset):

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
            invalidInputError(batch_size % node_num == 0,
                              "batch_size should be a multiple of num_shards, got"
                              " batch_size {}, node_num {}".format(batch_size, node_num))
            batch_per_shard = batch_size // node_num
            self.drop_remainder = True
        elif batch_per_thread > 0:
            batch_per_shard = batch_per_thread
            self.drop_remainder = False
        else:
            invalidInputError(False,
                              "one of batch_size or batch_per_thread must be larger than 0")

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
        invalidInputError(not self.drop_remainder,
                          "sanity check: drop_remainder should be false in this case,"
                          " otherwise please report a bug")
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

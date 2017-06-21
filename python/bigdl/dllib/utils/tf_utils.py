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

import tempfile

import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile

from bigdl.nn.layer import Model

def convert(input_ops, output_ops, sess):
    """
    Convert tensorflow model to bigdl model
    :param input_ops: operation list used for input, should be placeholders
    :param output_ops: operations list used for output
    :param sess: current tensorflow session
    :return: bigdl model
    """
    input_names = map(lambda x: x.name.split(":")[0], input_ops)
    output_names = map(lambda x: x.name.split(":")[0], output_ops)
    temp = tempfile.mkdtemp()

    saver = tf.train.Saver()
    saver.save(sess, temp + '/model.chkp')
    tf.train.write_graph(sess.graph, temp, 'model.pbtxt')

    merge_checkpoint(temp + '/model.pbtxt',
                     temp + '/model.chkp',
                     output_names,
                     temp + '/model.pb', sess)
    return Model.load_tensorflow(temp + '/model.pb', input_names, output_names)

def merge_checkpoint(input_graph,
                     checkpoint,
                     output_node_names,
                     output_graph,
                     sess):
    """
    Get the variable values from the checkpoint file, and merge them to the GraphDef file
    Args:
        input_graph: the GraphDef file, doesn't contain variable values
        checkpoint: the checkpoint file
        output_node_names: A list of string, the output names
        output_graph: String of the location and the name of the
            output graph
    """
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    input_graph_def = graph_pb2.GraphDef()
    with gfile.FastGFile(input_graph, "r") as f:
        text_format.Merge(f.read().decode("utf-8"), input_graph_def)

    for node in input_graph_def.node:
        node.device = ""

    importer.import_graph_def(input_graph_def, name="")

    sess.run([restore_op_name], {filename_tensor_name: checkpoint})
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names,
        variable_names_blacklist=""
    )

    with gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
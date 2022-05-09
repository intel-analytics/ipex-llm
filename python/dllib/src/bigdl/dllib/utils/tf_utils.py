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
import shutil

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from bigdl.dllib.nn.layer import Model
from bigdl.dllib.utils.common import JTensor
from bigdl.dllib.utils.common import callBigDlFunc
from bigdl.dllib.utils.log4Error import *
import os


def get_path(output_name, sess=None):
    if sess is None:
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

    temp = tempfile.mkdtemp()

    saver = tf.train.Saver()
    saver.save(sess, temp + '/model.chkp')
    tf.train.write_graph(sess.graph, temp, 'model.pbtxt')

    merge_checkpoint(temp + '/model.pbtxt',
                     temp + '/model.chkp',
                     [output_name],
                     temp + '/model.pb', sess)
    return temp + '/model.pb'


def convert(input_ops, output_ops, byte_order, bigdl_type):
    """
    Convert tensorflow model to bigdl model
    :param input_ops: operation list used for input, should be placeholders
    :param output_ops: operations list used for output
    :return: bigdl model
    """

    input_names = map(lambda x: x.name.split(":")[0], input_ops)
    output_names = map(lambda x: x.name.split(":")[0], output_ops)
    temp = tempfile.mkdtemp()

    dump_model(path=temp)
    model_path = temp + '/model.pb'
    bin_path = temp + '/model.bin'

    model = Model.load_tensorflow(model_path, input_names, output_names,
                                  byte_order, bin_path, bigdl_type)

    try:
        shutil.rmtree(temp)
    except OSError as e:
        if e.errno != errno.ENOENT:
            invalidOperationError(False, str(e), cause=e)

    return model


def export_checkpoint(checkpoint_path):
    """
    Export variable tensors from the checkpoint files.

    :param checkpoint_path: tensorflow checkpoint path
    :return: dictionary of tensor. The key is the variable name and the value is the numpy
    """
    reader = tf.train.NewCheckpointReader(checkpoint_path)

    # Get tensor name list
    tensor_names = filter(lambda n: n != 'global_step',
                          reader.get_variable_to_shape_map().keys())
    # Prepare key-value dictionary
    tensors = {}
    for tn in tensor_names:
        tensors[tn] = reader.get_tensor(tn)

    return tensors


def save_variable_bigdl(tensors, target_path, bigdl_type="float"):
    """
    Save a variable dictionary to a Java object file, so it can be read by BigDL

    :param tensors: tensor dictionary
    :param target_path: where is the Java object file store
    :param bigdl_type: model variable numeric type
    :return: nothing
    """
    import numpy as np
    jtensors = {}
    for tn in tensors.keys():
        if not isinstance(tensors[tn], np.ndarray):
            value = np.array(tensors[tn])
        else:
            value = tensors[tn]
        jtensors[tn] = JTensor.from_ndarray(value)

    callBigDlFunc(bigdl_type, "saveTensorDictionary", jtensors, target_path)


def dump_model(path, graph=None, sess=None, ckpt_file=None, bigdl_type="float"):
    """
    Dump a tensorflow model to files. The graph will be dumped to path/model.pb, and the checkpoint
    will be dumped to path/model.bin

    :param path: dump folder path
    :param sess: if user pass in session, we assume that the variable of the graph in the session
    has been inited
    :param graph: tensorflow graph. Default use the default graph of the session
    :param bigdl_type: model variable numeric type
    :return: nothing
    """
    if not os.path.isdir(path):
        invalidInputError(False, "Folder " + path + " does not exist")

    temp = None
    if ckpt_file is None:
        if sess is None:
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            temp = tempfile.mkdtemp()
            ckpt_file = temp
        # dump checkpoint to temp files
        saver = tf.train.Saver()
        saver.save(sess, ckpt_file)

    # generate bin files
    tensors = export_checkpoint(ckpt_file)
    save_variable_bigdl(tensors, path + "/model.bin", bigdl_type)

    # dump grap to pb file
    graph = sess.graph if graph is None else graph
    with gfile.GFile(path + "/model.pb", "wb") as f:
        f.write(graph.as_graph_def().SerializeToString())
    if temp is not None:
        try:
            shutil.rmtree(temp)
        except OSError as e:
            if e.errno != errno.ENOENT:
                invalidOperationError(False, str(e), cause=e)


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

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
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
import tensorflow as tf

def merge_checkpoint(input_graph,
                 input_checkpoint,
                 output_node_names,
                 output_graph):
    """
    merge the checkpoint file with the non-binary graph file to
    generate one GraphDef file with the variable values
    Args:
        input_graph: the GraphDef file, not in the binary form
        input_checkpoint: the checkpoint file
        output_node_names: A string of name of the output names, 
            use comma to seperate multi outputs
        output_graph: String of the location and the name of the
            output graph
    """
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    input_graph_def = graph_pb2.GraphDef()
    mode = "r"
    with gfile.FastGFile(input_graph, mode) as f:
        text_format.Merge(f.read().decode("utf-8"), input_graph_def)
    for node in input_graph_def.node:
      node.device = ""
    _ = importer.import_graph_def(input_graph_def, name="")
    with session.Session() as sess:
        sess.run([restore_op_name], {filename_tensor_name: input_checkpoint})
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names,
            variable_names_blacklist="")
    with gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

def run_model(end_points, output_path, model_scope=None):
    outputs = []
    results = []
    grad_inputs = []
    grad_vars = []
    grad_results = []
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope)
    i = 0
    opt = tf.train.GradientDescentOptimizer(0.01)
    for end_point in end_points:
        output = tf.Variable(tf.random_uniform(tf.shape(end_point)), name='output' + str(i))
        outputs.append(output)
        results.append(tf.assign(output, end_point, name = 'assign' + str(i)))

        # set up backward variables
        # filter None tensor
        tmp_vars = filter(lambda x: tf.gradients(end_point, x) != [None], trainable_vars)
        # set up random gradient input
        grad_input = tf.Variable(tf.random_uniform(tf.shape(end_point)), name='grad_input' + str(i))
        grad_inputs.append(grad_input)
        # compute gradients with random input
        backward = opt.compute_gradients(end_point, var_list=tmp_vars, grad_loss=grad_input)
        j = 0
        for gradients, tensor in backward:
            grad_var = tf.Variable(tf.random_uniform(tf.shape(tensor)), 
                name='{}_grad{}'.format(tensor.name[:-2], i))
            grad_vars.append(grad_var)
            grad_result = tf.assign(grad_var, gradients, name='grad_assign' + str((i+1)*j))
            grad_results.append(grad_result)
            j = j + 1
        i = i + 1

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init) 
        sess.run(results)
        sess.run(grad_results)
        saver.save(sess, output_path + '/model.chkp')
        tf.train.write_graph(sess.graph, output_path, 'model.pbtxt')
        # tf.summary.FileWriter(output_path + '/log', sess.graph)

    input_graph = output_path + "/model.pbtxt"    
    input_checkpoint = output_path + "/model.chkp"
    output_file = output_path + "/model.pb"

    output_nodes = map(lambda x: 'assign' + str(x), range(len(end_points)))
    grades_nodes = map(lambda x: 'grad_assign' + str(x), range(len(grad_results)))
    output_nodes.extend(grades_nodes)

    # merge_checkpoint(input_graph, input_checkpoint, map(lambda x: 'assign' + str(x), range(len(end_points))), output_file)
    merge_checkpoint(input_graph, input_checkpoint, output_nodes, output_file)


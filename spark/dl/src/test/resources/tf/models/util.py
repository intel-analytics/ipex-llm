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

import time

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

def run_model(end_points, output_path, model_scope=None, backward=True):
    grad_inputs = []
    grad_inputs_assign = []
    grad_vars = []
    grad_results = []

    if backward:
        loss = reduce(lambda x, y: tf.abs(x - y), end_points)
        loss = loss * loss
        for i in range(len(end_points)):
            grad_input = tf.Variable(tf.random_uniform(tf.shape(end_points[i]), minval=0.5, maxval=1),
                                     name='grad_input' + str(i))
            grad_inputs.append(grad_input)
            grad_input_endpoint = tf.gradients(loss, end_points[i])[0]
            grad_inputs_assign.append(tf.assign(grad_input, grad_input_endpoint, name = 'grad_input_assign' + str(i)))
        t = time.time()
        opt = tf.train.GradientDescentOptimizer(0.01)
        backward_vars = opt.compute_gradients(loss)
        tt = (time.time() - t) * 1000
        k = 0
        for gradients, tensor in backward_vars:
            if gradients is None:
                continue
            grad_var = tf.Variable(tf.random_uniform(tf.shape(tensor)),
                name='{}_grad'.format(tensor.name[:-2]))
            grad_vars.append(grad_var)
            grad_result = tf.assign(grad_var, gradients, name='grad_assign' + str(k))
            grad_results.append(grad_result)
            k = k + 1
        print 'Compute {} variables for backward in {} ms'.format(k, tt)

    saver = tf.train.Saver()
    output_results = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init) 
        tensorflow_tensors = sess.run(end_points)
        i = 0
        for e in end_points:
            tf.constant(tensorflow_tensors[i], name='output' + str(i))
            output_results.append('output' + str(i))
            i = i + 1

        if backward:
            sess.run(grad_results)
            sess.run(grad_inputs_assign)
        saver.save(sess, output_path + '/model.chkp')
        tf.train.write_graph(sess.graph, output_path, 'model.pbtxt')
        tf.summary.FileWriter('/tmp/testlog', sess.graph)

    input_graph = output_path + "/model.pbtxt"    
    input_checkpoint = output_path + "/model.chkp"
    output_file = output_path + "/model.pb"

    output_nodes = map(lambda x: x.name.split(":")[0], end_points)
    output_nodes.extend(output_results)
    if backward:
        grades_nodes = map(lambda x: 'grad_assign' + str(x), range(len(grad_results)))
        grades_input_nodes = map(lambda x: 'grad_input_assign' + str(x), range(len(grad_inputs)))
        output_nodes.extend(grades_nodes)
        output_nodes.extend(grades_input_nodes)

    merge_checkpoint(input_graph, input_checkpoint, output_nodes, output_file)


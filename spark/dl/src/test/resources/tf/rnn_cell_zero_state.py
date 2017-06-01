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
import numpy as np
import os
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops

def main():
    """
    Run this command to generate the pb file
    1. mkdir model
    2. python rnn_cell_zero_state.py
    3. wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
    4. python freeze_graph.py --input_graph model/rnn_cell.pbtxt --input_checkpoint model/rnn_cell.chkp --output_node_names=output --output_graph "rnn_cell.pb"
    """
    dir = os.path.dirname(os.path.realpath(__file__))
    n_input = 10
    n_hidden = 20

    xs = tf.placeholder(tf.float32, [None, n_input])
    W = tf.Variable(tf.constant(1.0, shape=[n_hidden, 5], dtype=tf.float32))
    b = tf.Variable(tf.constant(2.0, shape=[5], dtype=tf.float32))

    cell = rnn.BasicRNNCell(n_hidden)
    batch_size = array_ops.shape(xs)[0]
    state = cell.zero_state(batch_size, tf.float32)
    tf.nn.bias_add(tf.matmul(state, W), b, name="output")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter(dir + '/model/logs', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpointpath = saver.save(sess, dir + '/model/rnn_cell.chkp')
        tf.train.write_graph(sess.graph, dir + '/model', 'rnn_cell.pbtxt')
if __name__ == "__main__":
    main()
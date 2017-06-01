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

def main():
    """
    Run this command to generate the pb file
    1. mkdir model
    2. python rnn.py
    3. wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
    4. python freeze_graph.py --input_graph model/lstm.pbtxt --input_checkpoint model/lstm.chkp --output_node_names=output --output_graph "lstm.pb"
    """
    dir = os.path.dirname(os.path.realpath(__file__))
    n_steps = 5
    n_input = 10
    n_hidden = 20
    n_output = 5
    xs = tf.placeholder(tf.float32, [None, n_steps, n_input])
    weight = tf.Variable(tf.random_normal([n_hidden, n_output]), name="weight")
    bias = tf.Variable(tf.random_normal([n_output]), name="bias")

    x = tf.unstack(xs, n_steps, 1)

    cell = rnn.BasicLSTMCell(n_hidden)

    output, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    final = tf.nn.bias_add(tf.matmul(output[-1], weight), bias, name='output')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter(dir + '/model/logs', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpointpath = saver.save(sess, dir + '/model/lstm.chkp')
        tf.train.write_graph(sess.graph, dir + '/model', 'lstm.pbtxt')
if __name__ == "__main__":
    main()
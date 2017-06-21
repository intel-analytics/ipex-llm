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
from sys import argv
from tensorflow.contrib import rnn
from util import merge_checkpoint

def main():
    """
    Run this command to generate the pb file
    1. mkdir model
    2. python rnn_lstm.py
    """
    dir = argv[1]
    n_steps = 2
    n_input = 10
    n_hidden = 20
    n_output = 5
    # xs = tf.placeholder(tf.float32, [None, n_steps, n_input])
    xs = tf.Variable(tf.random_uniform([4, n_steps, n_input]) + 10, name='input', dtype=tf.float32)
    weight = tf.Variable(tf.random_uniform([n_hidden, n_output]) + 10, name="weight", dtype=tf.float32)
    bias = tf.Variable(tf.random_uniform([n_output]) + 10, name="bias", dtype=tf.float32)

    x = tf.unstack(xs, n_steps, 1)

    cell = rnn.BasicLSTMCell(n_hidden)

    output, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    final = tf.nn.bias_add(tf.matmul(output[-1], weight), bias, name='output')

    output = tf.Variable(tf.random_uniform(tf.shape(final)),name='output_result')
    result = tf.assign(output, final)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(result)
        checkpointpath = saver.save(sess, dir + '/model.chkp')
        tf.train.write_graph(sess.graph, dir, 'model.pbtxt')

    input_graph = dir + "/model.pbtxt"
    input_checkpoint = dir + "/model.chkp"
    output_node_names= ["output", "output_result"]
    output_graph = dir + "/model.pb"

    merge_checkpoint(input_graph, input_checkpoint, output_node_names, output_graph)
if __name__ == "__main__":
    main()

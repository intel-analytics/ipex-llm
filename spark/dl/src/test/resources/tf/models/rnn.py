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
from util import run_model

def main():
    """
    Run this command to generate the pb file
    1. mkdir model
    2. python rnn.py
    """
    tf.set_random_seed(1)
    n_steps = 2
    n_input = 10
    n_hidden = 20
    n_output = 5
    xs = tf.Variable(tf.random_uniform([4, n_steps, n_input]), name='input', dtype=tf.float32)
    xs = tf.identity(xs, "input_node")
    weight = tf.Variable(tf.random_uniform([n_hidden, n_output]), name="weight", dtype=tf.float32)
    bias = tf.Variable(tf.random_uniform([n_output]), name="bias", dtype=tf.float32)
    x = tf.unstack(xs, n_steps, 1)
    cell = rnn.BasicRNNCell(n_hidden)
    output, states = rnn.static_rnn(cell, x, dtype=tf.float32)
    final = tf.nn.bias_add(tf.matmul(output[-1], weight), bias, name='output')

    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], 'rnn', argv[3] == 'True')
if __name__ == "__main__":
    main()

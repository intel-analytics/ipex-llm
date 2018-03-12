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

from util import run_model

def main():
    tf.set_random_seed(10)
    with tf.Session() as sess:
        rnn_cell = tf.nn.rnn_cell.LSTMCell(10)

        # defining initial state
        initial_state = rnn_cell.zero_state(4, dtype=tf.float32)

        inputs = tf.Variable(tf.random_uniform(shape = (4, 30, 100)), name='input')
        inputs = tf.identity(inputs, "input_node")

        # 'state' is a tensor of shape [batch_size, cell_state_size]
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=initial_state, dtype=tf.float32)

        y1 = tf.identity(outputs, 'outputs')
        y2 = tf.identity(state, 'state')

        t1 = tf.ones([4, 30, 10])
        t2 = tf.ones([4, 10])

        loss = tf.reduce_sum((y1 - t1) * (y1 - t1)) + tf.reduce_sum((y2 - t2) * (y2 - t2))
        tf.identity(loss, name = "lstm_loss")
        grad = tf.identity(tf.gradients(loss, inputs), name='gradOutput')
        # tf.summary.FileWriter('/tmp/log', tf.get_default_graph())

        net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
        run_model(net_outputs, argv[1], None, argv[3] == 'True')

if __name__ == "__main__":
    main()

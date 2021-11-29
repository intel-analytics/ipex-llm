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
from util import merge_checkpoint

def main():
    """
    Run this command to generate the pb file
    1. mkdir model
    2. python test.py
    3. wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
    4. python freeze_graph.py --input_graph model/share_weight.pbtxt --input_checkpoint model/share_weight.chkp --output_node_names=output --output_graph "share_weight.pb"
    """
    xs = tf.placeholder(tf.float32, [None, 10])
    W1 = tf.Variable(tf.random_normal([10,10]))
    b1 = tf.Variable(tf.random_normal([10]))
    Wx_plus_b1 = tf.nn.bias_add(tf.matmul(xs,W1), b1)
    output= tf.nn.tanh(Wx_plus_b1)

    Wx_plus_b2 = tf.nn.bias_add(tf.matmul(output,W1), b1)
    W2 = tf.Variable(tf.random_normal([10, 1]))
    b2 = tf.Variable(tf.random_normal([1]))
    final = tf.nn.bias_add(tf.matmul(Wx_plus_b2, W2), b2, name='output')
    dir = argv[1]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpointpath = saver.save(sess, dir + '/model.chkp')
        tf.train.write_graph(sess.graph, dir, 'model.pbtxt')

    input_graph = dir + "/model.pbtxt"
    input_checkpoint = dir + "/model.chkp"
    output_node_names = "output"
    output_graph = dir + "/model.pb"

    merge_checkpoint(input_graph, input_checkpoint, [output_node_names], output_graph)

if __name__ == "__main__":
    main()

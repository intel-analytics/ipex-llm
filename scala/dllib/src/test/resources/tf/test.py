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

def main():
    """
    Run this command to generate the pb file
    1. mkdir model
    2. python test.py
    3. wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
    4. python freeze_graph.py --input_graph model/test.pbtxt --input_checkpoint model/test.chkp --output_node_names=output --output_graph "test.pb"
    """
    dir = os.path.dirname(os.path.realpath(__file__))
    xs = tf.placeholder(tf.float32, [None, 1])
    W1 = tf.Variable(tf.zeros([1,10])+0.2)
    b1 = tf.Variable(tf.zeros([10])+0.1)
    Wx_plus_b1 = tf.nn.bias_add(tf.matmul(xs,W1), b1)
    output= tf.nn.tanh(Wx_plus_b1)

    W2 = tf.Variable(tf.zeros([10,1])+0.2)
    b2 = tf.Variable(tf.zeros([1])+0.1)
    Wx_plus_b2 = tf.nn.bias_add(tf.matmul(output,W2), b2, name='output')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpointpath = saver.save(sess, dir + '/model/test.chkp')
        tf.train.write_graph(sess.graph, dir + '/model', 'test.pbtxt')
if __name__ == "__main__":
    main()

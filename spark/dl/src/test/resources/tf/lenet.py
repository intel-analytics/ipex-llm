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
import slim.nets.lenet as lenet

def main():
    """
    Run this command to generate the pb file
    1. git clone https://github.com/tensorflow/models.git
    2. export PYTHONPATH=Path_to_your_model_folder
    1. mkdir model
    2. python lenet.py
    3. wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
    4. python freeze_graph.py --input_graph model/lenet.pbtxt --input_checkpoint model/lenet.chkp --output_node_names="LeNet/fc4/BiasAdd" --output_graph lenet.pb
    """
    dir = os.path.dirname(os.path.realpath(__file__))
    batch_size = 5
    height, width = 32, 32
    inputs = tf.placeholder(tf.float32, [None, height, width, 3])
    net, end_points  = lenet.lenet(inputs)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpointpath = saver.save(sess, dir + '/model/lenet.chkp')
        tf.train.write_graph(sess.graph, dir + '/model', 'lenet.pbtxt')
        tf.summary.FileWriter(dir + '/log', sess.graph)
if __name__ == "__main__":
    main()

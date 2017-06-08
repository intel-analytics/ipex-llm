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
import slim.nets.inception_resnet_v2 as inception_resnet_v2

import merge_checkpoint as merge



def main():
    """
    You can run these commands to generate the pb file
    1. git clone https://github.com/tensorflow/models.git
    2. export PYTHONPATH=Path_to_your_model_folder, eg. /home/models/
    3. python inception_resnet_v2.py
    """
    dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(dir + '/model'):
        os.mkdir(dir + '/model')
    batch_size = 5
    height, width = 299, 299
    num_classes = 1001
    # inputs = tf.placeholder(tf.float32, [None, height, width, 3])
    inputs = tf.Variable(tf.random_uniform((2, height, width, 3)), name='input')
    net, end_points = inception_resnet_v2.inception_resnet_v2(inputs,is_training = False)
    output1 = tf.Variable(tf.random_uniform(tf.shape(net)),name='output1')
    result1 = tf.assign(output1,net)
    output2 = tf.Variable(tf.random_uniform(tf.shape(end_points['AuxLogits'])),name='output2')
    result2 = tf.assign(output2,end_points['AuxLogits'])
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run([result1,result2])
        checkpointpath = saver.save(sess, dir + '/model/inception_resnet_v2.chkp')
        tf.train.write_graph(sess.graph, dir + '/model', 'inception_resnet_v2.pbtxt')
        tf.summary.FileWriter(dir + '/log', sess.graph)


    input_graph = dir + "/model/inception_resnet_v2.pbtxt"    
    input_checkpoint = dir + "/model/inception_resnet_v2.chkp"
    output_node_names= "InceptionResnetV2/AuxLogits/Logits/BiasAdd,InceptionResnetV2/Logits/Logits/BiasAdd,output1,output2"
    output_graph = dir + "/inception_resnet_v2_save.pb"
    
    merge.merge_checkpoint(input_graph, input_checkpoint, output_node_names, output_graph)

if __name__ == "__main__":
    main()

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
# Part of the code is reference https://github.com/tensorflow/models/blob/master/slim/nets/lenet.py
#

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def main():
    inputs = tf.placeholder(tf.float32, shape=(1, 1, 28, 28))
    net, end_points  = lenet(inputs)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.save(sess, 'model/model.chkp')
        tf.train.write_graph(sess.graph, 'model', 'model.pbtxt')

def lenet(images):
    end_points = {}
    num_classes=10
    is_training=False
    dropout_keep_prob=0.5
    prediction_fn=slim.softmax
    scope='LeNet'

    with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1', data_format="NCHW")
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1', data_format="NCHW")
        net = slim.conv2d(net, 64, [5, 5], scope='conv2', data_format="NCHW")
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2', data_format="NCHW")
        net = slim.flatten(net)
        end_points['Flatten'] = net

        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='fc4')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points

if __name__ == "__main__":
    main()
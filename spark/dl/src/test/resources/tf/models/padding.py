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
from sys import argv

from util import run_model

def main():
    """
    You can also run these commands manually to generate the pb file
    1. git clone https://github.com/tensorflow/models.git
    2. export PYTHONPATH=Path_to_your_model_folder
    3. python padding.py
    """
    height, width = 28, 28
    inputs = tf.Variable(tf.random_uniform((2, height, width, 3), minval=1, maxval=2), name='input')
    W = tf.Variable(tf.random_uniform([3, 3, 3, 1], minval=1, maxval=2), name="weight")
    b = tf.Variable(tf.random_uniform([1], minval=1, maxval=2), name="bias")
    conv = tf.nn.conv2d(inputs, W, strides=[1, 2, 2, 1], padding="SAME", name="conv")
    output = tf.nn.bias_add(conv, b, name="output")

    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], 'InceptionResnetV2', argv[3] == 'True')

if __name__ == "__main__":
    main()
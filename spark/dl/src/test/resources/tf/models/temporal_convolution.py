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
    3. python temporal_convolution.py
    """
    tf.set_random_seed(1024)
    input_width = 32
    input_channel = 3
    inputs = tf.Variable(tf.random_uniform((1, input_width, input_channel)), name='input')
    inputs = tf.identity(inputs, "input_node")
    filter_width = 4
    output_channels = 6
    filters = tf.Variable(tf.random_uniform((filter_width, input_channel, output_channels)))
    conv_out = tf.nn.conv1d(inputs, filters, stride=1, padding="VALID")
    bias = tf.Variable(tf.zeros([output_channels]))

    output = tf.nn.tanh(tf.nn.bias_add(conv_out, bias), name="output")

    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], backward=(argv[3] == 'True'))

if __name__ == "__main__":
    main()
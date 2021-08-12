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

    inputs = tf.Variable(tf.reshape(tf.range(0.0, 16), [1, 1, 4, 4]), name = 'input')
    inputs = tf.identity(inputs, "input_node")
    output = tf.layers.batch_normalization(inputs, axis=1, training=True)

    named_output = tf.nn.relu(output, name="output")

    net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
    run_model(net_outputs, argv[1], 'batchNorm', argv[3] == 'True')

if __name__ == "__main__":
    main()
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
        inputs = tf.Variable(tf.random_uniform((20, 30, 32)), name = 'input')
        inputs = tf.identity(inputs, "input_node")

        input1, input2, input3, input4 = tf.split(inputs, 4, 0)
        # scatter and gather
        tensor_array = tf.TensorArray(tf.float32, 128)
        tensor_array = tensor_array.scatter([1, 2, 5, 4, 3], input1)
        tensor_array.gather([1, 2, 5, 4, 3], name='scatter_and_gather')

        # split and concat
        tensor_array = tf.TensorArray(tf.float32, 2)
        tensor_array = tensor_array.split(input2, [2, 3])
        tf.identity(tensor_array.concat(), name='split_and_concat')

        # write and read
        tensor_array = tf.TensorArray(tf.float32, 5)
        tensor_array = tensor_array.identity()
        tensor_array = tensor_array.write(1, input3)
        tf.cast(tensor_array.size(), tf.float32, name='size1')
        tensor_array.read(1, name='write_and_read')
        tf.cast(tensor_array.size(), tf.float32, name='size2')

        # unstack and stack
        tensor_array = tf.TensorArray(tf.float32, 5)
        tensor_array = tensor_array.unstack(input4)
        tf.identity(tensor_array.stack(), name='unstack_and_stack')

        net_outputs = map(lambda x: tf.get_default_graph().get_tensor_by_name(x), argv[2].split(','))
        run_model(net_outputs, argv[1], None, argv[3] == 'True')

if __name__ == "__main__":
    main()

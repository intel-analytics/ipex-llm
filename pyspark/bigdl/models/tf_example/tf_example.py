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
from bigdl.util.tf_utils import convert

def main():
    input = tf.placeholder(tf.float32, [None, 5])
    weight = tf.Variable(tf.random_uniform([5, 10]))
    bias = tf.Variable(tf.random_uniform([10]))
    middle = tf.nn.bias_add(tf.matmul(input, weight), bias)
    output= tf.nn.tanh(middle)

    tensor = np.random.rand(5, 5)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        tensorflow_result = sess.run(output, {input: tensor})
        bigdl_model = convert([input], [output], sess)
        bigdl_result = bigdl_model.forward(tensor)[0]

        print("Tensorflow forward result is " + str(tensorflow_result))
        print("BigDL forward result is " + str(bigdl_result))

        np.testing.assert_almost_equal(tensorflow_result, bigdl_result, 6)
        print("The results are almost equal in 6 decimals")


if __name__ == "__main__":
    main()
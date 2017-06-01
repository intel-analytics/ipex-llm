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
from tensorflow.python.platform import gfile
import os

def main():
    dir = os.path.dirname(os.path.realpath(__file__))
    xs = tf.placeholder(tf.float32, [4, 3, 4])
    tf.nn.relu(xs)

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph, dir, 'model.pb')

if __name__ == "__main__":
    main()

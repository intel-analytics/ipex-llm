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
from tensorflow.python.platform import gfile

def main():
    with gfile.FastGFile("/tmp/tensorflow9012247417719342743saver",'rb') as f:
        graph_def = tf.GraphDef()
        content = f.read()
        graph_def.ParseFromString(content)
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            sess = tf.Session()
            for op in graph.get_operations():
                print(op.name)
            prediction = graph.get_tensor_by_name('relu:0')
            ix = graph.get_tensor_by_name('input:0')
            rand_array = np.random.rand(2, 4)
            print(sess.run(prediction, feed_dict={ix: rand_array}))

if __name__ == "__main__":
    main()

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
from sys import argv

def main():
    with gfile.FastGFile(argv[1],'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            sess = tf.Session()
            output_suffix = ''
            if len(argv) == 3:
                output_suffix = argv[2]
            output = graph.get_tensor_by_name('output' + output_suffix + ':0')
            target = graph.get_tensor_by_name('target:0')
            tf_output = sess.run(output)
            bigdl_output = sess.run(target)
            print("Tensorflow output is:")
            print(tf_output)
            print("BigDL output is:")
            print(bigdl_output)
            np.testing.assert_almost_equal(tf_output, bigdl_output, 4)

if __name__ == "__main__":
    main()

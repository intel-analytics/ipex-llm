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
from tensorflow.python.platform import gfile
from sys import argv

file = argv[1]
output = argv[2]
output_folder = argv[3]
output_file = argv[4]
output_tensor_name = argv[5]

with gfile.FastGFile(file,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            output_node = graph.get_tensor_by_name(output)
            result = sess.run(output_node)
            result_node = tf.constant(result, name = output_tensor_name)
            tf.train.write_graph(sess.graph_def, output_folder, output_file, False)

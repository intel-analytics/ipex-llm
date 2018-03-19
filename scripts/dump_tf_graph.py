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

#
# This scripts can dump tensorflow protobuf graph to tensorflow summary format. So user can see
# the graph in the tensorboard.
#
# How to run this script:
# python dump_tf_graph.py protxt_file_path [log_dir]
#
# the default value of log_dir is ./log
#

from sys import argv
from tensorflow.python.platform import gfile
import tensorflow as tf

def main():
    if len(argv) == 1 or len(argv) > 3:
        print("How to run this script:")
        print("python dump_tf_graph.py protxt_file_path [log_dir]")
        exit(1)

    log_dir = 'log'
    if len(argv) == 3:
        log_dir = argv[2]

    with gfile.FastGFile(argv[1],'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            tf.summary.FileWriter(log_dir, graph)

if __name__ == "__main__":
    main()
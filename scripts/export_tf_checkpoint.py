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
from sys import argv
from bigdl.util.tf_utils import dump_model

import tensorflow as tf

def main():
    """
    How to run this script:
    python export_tf_checkpoint.py meta_file chkp_file save_path
    """
    saver = tf.train.import_meta_graph(argv[1])
    with tf.Session() as sess:
        saver.restore(sess, argv[2])
        dump_model(argv[3], sess)

if __name__ == "__main__":
    main()
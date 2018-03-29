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
# This script can export tensorflow graph and variables from tensorflow checkpoint files or
# tf.saved_model API saved folder. The graph file is named as model.pb, which is protobuf format.
# Varibales file is named as model.bin, which is loadable by BigDL.
#
# How to run this script:
# python export_tf_checkpoint.py checkpoint_name
# python export_tf_checkpoint.py saver_folder
# python export_tf_checkpoint.py checkpoint_name save_path
# python export_tf_checkpoint.py saver_folder save_path
# python export_tf_checkpoint.py meta_file checkpoint_name save_path
#
# the default value of save_path is model
#

from sys import argv
from bigdl.util.tf_utils import dump_model

import tensorflow as tf
import os.path as op
import os

def main():
    meta_file = None
    checkpoint = None
    save_path = "model"
    saver_folder = None

    if len(argv) == 2:
        if op.isdir(argv[1]):
            saver_folder = argv[1]
        else:
            meta_file = argv[1] + ".meta"
            checkpoint = argv[1]
    elif len(argv) == 3:
        if op.isdir(argv[1]):
            saver_folder = argv[1]
        else:
            meta_file = argv[1] + ".meta"
            checkpoint = argv[1]
        save_path = argv[2]
    elif len(argv) == 4:
        meta_file = argv[1]
        checkpoint = argv[2]
        save_path = argv[3]
    else:
        print("Invalid script arguments. How to run the script:\n" +
              "python export_tf_checkpoint.py checkpoint_name\n" +
              "python export_tf_checkpoint.py saver_folder\n" +
              "python export_tf_checkpoint.py checkpoint_name save_path\n" +
              "python export_tf_checkpoint.py saver_folder save_path\n" +
              "python export_tf_checkpoint.py meta_file checkpoint_name save_path")
        exit(1)

    if op.isfile(save_path):
        print("The save folder is a file. Exit")
        exit(1)

    if not op.exists(save_path):
        print("create folder " + save_path)
        os.makedirs(save_path)

    with tf.Session() as sess:
        if saver_folder is None:
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
            saver.restore(sess, checkpoint)
        else:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saver_folder)
            checkpoint = save_path + '/model.ckpt'
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
        dump_model(save_path, None, sess, checkpoint)

if __name__ == "__main__":
    main()
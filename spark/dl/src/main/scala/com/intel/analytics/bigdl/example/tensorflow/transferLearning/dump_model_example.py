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
from datasets import flowers
from inception_model import InceptionModel


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/inception_finetuned/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'data_dir', '/tmp/flowers/',
    'Directory contains the flowers data.')

tf.app.flags.DEFINE_string(
    'checkpoint_file_path', '/tmp/checkpoints/inception_v1.ckpt',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'dump_model_path', '/tmp/tfmodel/',
    'Directory where graph file and variable bin file are written to.'
)

tf.app.flags.DEFINE_string(
    'data_split', 'train',
    'Directory where graph file and variable bin file are written to.'
)

FLAGS = tf.app.flags.FLAGS


def main(_):

    # 1. change the dataset
    # dataset = imagenet.get_split('train'), FLAGS.data_dir)
    dataset = flowers.get_split(FLAGS.data_split, FLAGS.data_dir)

    model = InceptionModel(checkpoints_file=FLAGS.checkpoint_file_path)

    # 2. set the model to training mode
    # op, graph model.build(dataset, image_height=224, image_width=224, num_classes=1000, is_training=True)
    op, graph = model.build(dataset, image_height=224, image_width=224, num_classes=1000, is_training=False)

    # 3. comment out the actual training code
    # slim.learning.train(
    #     op,
    #     logdir=train_dir,
    #     init_fn=model.init_fn,
    #     number_of_steps=100)

    # 4. dump model to the specified path
    from bigdl.util.tf_utils import dump_model
    dump_model(path=FLAGS.dump_model_path, ckpt_file=FLAGS.checkpoint_file_path, graph=graph)

if __name__ == '__main__':
  tf.app.run()


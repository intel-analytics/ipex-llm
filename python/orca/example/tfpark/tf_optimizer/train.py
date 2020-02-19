#
# Copyright 2018 Analytics Zoo Authors.
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
from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import numpy as np
import sys

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=labels.dtype)
    is_correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(is_correct)


def main(max_epoch, data_num):
    sc = init_nncontext()

    # get data, pre-process and create TFDataset
    (train_images_data, train_labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    (test_images_data, test_labels_data) = mnist.read_data_sets("/tmp/mnist", "train")

    train_images_data = (train_images_data[:data_num] - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    train_labels_data = train_labels_data[:data_num].astype(np.int)
    test_images_data = (test_images_data[:data_num] - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    test_labels_data = (test_labels_data[:data_num]).astype(np.int)
    dataset = TFDataset.from_ndarrays((train_images_data, train_labels_data),
                                      batch_size=360,
                                      val_tensors=(test_images_data, test_labels_data))

    # construct the model from TFDataset
    images, labels = dataset.tensors

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    acc = accuracy(logits, labels)

    # create a optimizer
    optimizer = TFOptimizer.from_loss(loss, Adam(1e-3),
                                      metrics={"acc": acc},
                                      model_dir="/tmp/lenet/")
    # kick off training
    optimizer.optimize(end_trigger=MaxEpoch(max_epoch))

    saver = tf.train.Saver()
    saver.save(optimizer.sess, "/tmp/lenet/model")


if __name__ == '__main__':

    max_epoch = 5
    data_num = 60000

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
        data_num = int(sys.argv[2])
    main(max_epoch, data_num)

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
from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.tfpark import TFOptimizer, TFDataset
from bigdl.dllib.optim.optimizer import *
import numpy as np

from bigdl.dllib.feature.dataset import mnist
from bigdl.dllib.feature.dataset.transformer import *

import os
import sys
import argparse

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim

parser = argparse.ArgumentParser(description="Run the tfpark keras "
                                             "dataset example.")
parser.add_argument('--max_epoch', type=int, default=5,
                    help='Set max_epoch for training, it should be integer.')
parser.add_argument('--data_num', type=int, default=60000,
                    help='Set data_num for training, it should be integer.')                    
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local, yarn or spark-submit.')

def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=labels.dtype)
    is_correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(is_correct)


def main(max_epoch, data_num):
    args = parser.parse_args()
    cluster_mode = args.cluster_mode
    if cluster_mode.startswith("yarn"):
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        assert hadoop_conf, "Directory path to hadoop conf not found for yarn-client mode. Please " \
                "set the environment variable HADOOP_CONF_DIR"
        spark_conf = create_spark_conf().set("spark.executor.memory", "5g") \
            .set("spark.executor.cores", 2) \
            .set("spark.executor.instances", 2) \
            .set("spark.driver.memory", "2g")
        if cluster_mode == "yarn-client":
            sc = init_nncontext(spark_conf, cluster_mode="yarn-client", hadoop_conf=hadoop_conf)
        else:
            sc = init_nncontext(spark_conf, cluster_mode="yarn-cluster", hadoop_conf=hadoop_conf)
    else:
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

    args = parser.parse_args()
    max_epoch = args.max_epoch
    data_num = args.data_num

    main(max_epoch, data_num)

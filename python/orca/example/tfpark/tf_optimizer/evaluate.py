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
from optparse import OptionParser
import tensorflow as tf
from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.tfpark import TFDataset, TFPredictor
from bigdl.dllib.utils.common import *

import numpy as np
import sys
import os

from bigdl.dllib.feature.dataset import mnist

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim


def main(options, data_num):

    data_path = '/tmp/mnist' if not options.data_path else options.data_path
    cluster_mode = options.cluster_mode
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
    (images_data, labels_data) = mnist.read_data_sets(data_path, "test")
    images_data = (images_data[:data_num] - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    labels_data = labels_data[:data_num].astype(np.int32)
    dataset = TFDataset.from_ndarrays((images_data, labels_data), batch_per_thread=20)

    # construct the model from TFDataset
    images, labels = dataset.tensors

    labels = tf.squeeze(labels)

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=False)

    predictions = tf.to_int32(tf.argmax(logits, axis=1))
    correct = tf.expand_dims(tf.to_int32(tf.equal(predictions, labels)), axis=1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/tmp/lenet/model")

        predictor = TFPredictor(sess, [correct])

        accuracy = predictor.predict().mean()

        print("predict accuracy is %s" % accuracy)


if __name__ == '__main__':

    data_num = 10000

    if len(sys.argv) > 1:
        data_num = int(sys.argv[1])

    parser = OptionParser()
    parser.add_option("--data_path", dest="data_path")
    parser.add_option('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local, yarn or spark-submit.')
    (options, args) = parser.parse_args(sys.argv)

    main(options, data_num)

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
import argparse
import os

import tensorflow as tf
import numpy as np
from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.tfpark import KerasModel, TFDataset
from bigdl.dllib.utils.common import *

parser = argparse.ArgumentParser(description="Run the tfpark keras "
                                             "dataset example.")
parser.add_argument('--data_path', type=str, default='/tmp/mnist',
                    help='training data path.')                                             
parser.add_argument('--max_epoch', type=int, default=5,
                    help='Set max_epoch for training, it should be integer.')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local, yarn or spark-submit.')

def get_data_rdd(dataset, sc):
    data_path = args.data_path
    from bigdl.dllib.feature.dataset import mnist
    (images_data, labels_data) = mnist.read_data_sets(data_path, dataset)
    image_rdd = sc.parallelize(images_data)
    labels_rdd = sc.parallelize(labels_data)
    rdd = image_rdd.zip(labels_rdd) \
        .map(lambda rec_tuple: ((rec_tuple[0] - mnist.TRAIN_MEAN) / mnist.TRAIN_STD,
                                np.array(rec_tuple[1])))
    return rdd

def main(max_epoch):
    args = parser.parse_args()
    cluster_mode = args.cluster_mode
    if cluster_mode.startswith("yarn"):
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        assert hadoop_conf, "Directory path to hadoop conf not found for yarn-client mode. Please " \
                "set the environment variable HADOOP_CONF_DIR"
        spark_conf = create_spark_conf().set("spark.executor.memory", "5g") \
            .set("spark.executor.cores", 2) \
            .set("spark.executor.instances", 2) \
            .set("spark.executorEnv.HTTP_PROXY", "http://child-prc.intel.com:913") \
            .set("spark.executorEnv.HTTPS_PROXY", "http://child-prc.intel.com:913") \
            .set("spark.driver.memory", "2g")
        if cluster_mode == "yarn-client":
            sc = init_nncontext(spark_conf, cluster_mode="yarn-client", hadoop_conf=hadoop_conf)
        else:
            sc = init_nncontext(spark_conf, cluster_mode="yarn-cluster", hadoop_conf=hadoop_conf)
    else:
        sc = init_nncontext()

    training_rdd = get_data_rdd("train", sc)
    testing_rdd = get_data_rdd("test", sc)

    dataset = TFDataset.from_rdd(training_rdd,
                                 features=(tf.float32, [28, 28, 1]),
                                 labels=(tf.int32, []),
                                 batch_size=320,
                                 val_rdd=testing_rdd)

    model = tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    keras_model = KerasModel(model)

    keras_model.fit(dataset,
                    epochs=max_epoch,
                    distributed=True)

    eval_dataset = TFDataset.from_rdd(
        testing_rdd,
        features=(tf.float32, [28, 28, 1]),
        labels=(tf.int32, []), batch_per_thread=80)
    result = keras_model.evaluate(eval_dataset)

    print(result)
    # >> [0.08865142822265625, 0.9722]

    # the following assert is used for internal testing
    assert result['acc Top1Accuracy'] > 0.95

    model.save_weights("/tmp/mnist_keras.h5")


if __name__ == '__main__':

    args = parser.parse_args()
    max_epoch = args.max_epoch

    main(max_epoch)

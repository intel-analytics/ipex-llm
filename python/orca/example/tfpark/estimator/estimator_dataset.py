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
import numpy as np
from zoo import init_nncontext
from zoo.tfpark import TFDataset, TFEstimator
from zoo.tfpark import ZooOptimizer


def get_data(dataset):
    from bigdl.dataset import mnist
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
    images_data = (images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    return (images_data, labels_data.astype(np.int32))


def main():
    sc = init_nncontext()

    def model_fn(features, labels, mode):
        from nets import lenet
        slim = tf.contrib.slim
        with slim.arg_scope(lenet.lenet_arg_scope()):
            logits, end_points = lenet.lenet(features, num_classes=10, is_training=True)

        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.reduce_mean(
                tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

            optimizer = ZooOptimizer(tf.train.AdamOptimizer())
            train_op = optimizer.minimize(loss)
            return tf.estimator.EstimatorSpec(mode, predictions=logits,
                                              loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, predictions=logits)

    def input_fn(mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            training_data = get_data("train")
            dataset = TFDataset.from_ndarrays(training_data, batch_size=320)
        elif mode == tf.estimator.ModeKeys.EVAL:
            testing_data = get_data("test")
            dataset = TFDataset.from_ndarrays(testing_data, batch_per_thread=80)
        else:
            images, _ = get_data("test")
            dataset = TFDataset.from_ndarrays(images, batch_per_thread=80)

        return dataset
    estimator = TFEstimator.from_model_fn(model_fn, model_dir="/tmp/estimator")

    estimator.train(input_fn, steps=10)

    metrics = estimator.evaluate(input_fn, ["acc"])
    print(metrics)

    predictions = estimator.predict(input_fn)

    print(predictions.first())
    print("finished...")
    sc.stop()


if __name__ == '__main__':
    main()

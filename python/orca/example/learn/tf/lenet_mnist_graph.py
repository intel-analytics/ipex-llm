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
import argparse

import tensorflow as tf
from zoo.orca.learn.tf.estimator import Estimator
from zoo.orca import init_orca_context, stop_orca_context


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=labels.dtype)
    is_correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(is_correct)


def lenet(images):
    with tf.variable_scope('LeNet', [images]):
        net = tf.layers.conv2d(images, 32, (5, 5), activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, (2, 2), 2, name='pool1')
        net = tf.layers.conv2d(net, 64, (5, 5), activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, (2, 2), 2, name='pool2')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc3')
        logits = tf.layers.dense(net, 10)
        return logits


def preprocess(x, y):
    return tf.to_float(tf.reshape(x, (28, 28, 1))) / 255.0, y


def main(max_epoch):

    (train_feature, train_label), (val_feature, val_label) = tf.keras.datasets.mnist.load_data()

    # tf.data.Dataset.from_tensor_slices is for demo only. For production use, please use
    # file-based approach (e.g. tfrecord).
    train_dataset = tf.data.Dataset.from_tensor_slices((train_feature, train_label))
    train_dataset = train_dataset.map(preprocess)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_feature, val_label))
    val_dataset = val_dataset.map(preprocess)

    # tensorflow inputs
    images = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    # tensorflow labels
    labels = tf.placeholder(dtype=tf.int32, shape=(None,))

    logits = lenet(images)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    acc = accuracy(logits, labels)

    # create an estimator
    est = Estimator.from_graph(inputs=images,
                               outputs=logits,
                               labels=labels,
                               loss=loss,
                               optimizer=tf.train.AdamOptimizer(),
                               metrics={"acc": acc})
    est.fit(data=train_dataset,
            batch_size=320,
            epochs=max_epoch,
            validation_data=val_dataset)

    result = est.evaluate(val_dataset)
    print(result)

    est.save_tf_checkpoint("/tmp/lenet/model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')

    args = parser.parse_args()
    if args.cluster_mode == "local":
        init_orca_context(cluster_mode="local", cores=4)
    elif args.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="6g")
    main(5)
    stop_orca_context()

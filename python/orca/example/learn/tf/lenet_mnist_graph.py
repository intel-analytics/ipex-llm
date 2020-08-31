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
import tensorflow_datasets as tfds
import sys

from zoo.orca.learn.tf.estimator import Estimator
from zoo.orca import init_orca_context, stop_orca_context

sys.path.append("/tmp/models/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=labels.dtype)
    is_correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(is_correct)


def main(max_epoch):
    sc = init_orca_context(cluster_mode="local", cores=4, memory="2g")

    # get DataSet
    mnist_train = tfds.load(name="mnist", split="train")
    mnist_test = tfds.load(name="mnist", split="test")

    # Normalizes images
    def normalize_img(data):
        data['image'] = tf.cast(data["image"], tf.float32) / 255.
        return data

    mnist_train = mnist_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mnist_test = mnist_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # tensorflow inputs
    images = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    # tensorflow labels
    labels = tf.placeholder(dtype=tf.int32, shape=(None,))

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    acc = accuracy(logits, labels)

    # create an estimator
    est = Estimator.from_graph(inputs=images,
                               outputs=logits,
                               labels=labels,
                               loss=loss,
                               optimizer=tf.train.AdamOptimizer(),
                               metrics={"acc": acc})
    est.fit(data=mnist_train,
            batch_size=320,
            epochs=max_epoch,
            validation_data=mnist_test)

    result = est.evaluate(mnist_test)
    print(result)

    est.save_tf_checkpoint("/tmp/lenet/model")
    stop_orca_context()


if __name__ == '__main__':
    max_epoch = 5

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
    main(max_epoch)

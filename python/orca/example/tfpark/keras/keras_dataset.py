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
import sys

import tensorflow as tf
import numpy as np
from zoo import init_nncontext
from zoo.tfpark import KerasModel, TFDataset


def get_data_rdd(dataset, sc):
    from bigdl.dataset import mnist
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
    image_rdd = sc.parallelize(images_data)
    labels_rdd = sc.parallelize(labels_data)
    rdd = image_rdd.zip(labels_rdd) \
        .map(lambda rec_tuple: ((rec_tuple[0] - mnist.TRAIN_MEAN) / mnist.TRAIN_STD,
                                np.array(rec_tuple[1])))
    return rdd


def main(max_epoch):
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

    print(model.metrics_names)
    print(result)
    # >> ['loss', 'acc']
    # >> [0.08865142822265625, 0.9722]

    model.save_weights("/tmp/mnist_keras.h5")


if __name__ == '__main__':

    max_epoch = 5

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
    main(max_epoch)

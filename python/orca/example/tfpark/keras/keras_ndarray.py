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
from zoo import init_nncontext
from bigdl.dataset import mnist
from zoo.tfpark import KerasModel


def main(max_epoch):
    _ = init_nncontext()

    (training_images_data, training_labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    (testing_images_data, testing_labels_data) = mnist.read_data_sets("/tmp/mnist", "test")

    training_images_data = (training_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    testing_images_data = (testing_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD

    model = tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    keras_model = KerasModel(model)

    keras_model.fit(training_images_data,
                    training_labels_data,
                    validation_data=(testing_images_data, testing_labels_data),
                    epochs=max_epoch,
                    batch_size=320,
                    distributed=True)

    result = keras_model.evaluate(testing_images_data, testing_labels_data,
                                  distributed=True, batch_per_thread=80)

    print(result)
    # >> [0.08865142822265625, 0.9722]

    # the following assert is used for internal testing
    assert result['acc Top1Accuracy'] > 0.95

    keras_model.save_weights("/tmp/mnist_keras.h5")


if __name__ == '__main__':

    max_epoch = 5

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
    main(max_epoch)

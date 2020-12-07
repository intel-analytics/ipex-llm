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
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf2 import Estimator


def preprocess(x, y):
    x = tf.cast(tf.reshape(x, (28, 28, 1)), dtype=tf.float32) / 255.0
    return x, y


def train_data_creator(config):
    (train_feature, train_label), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((train_feature, train_label))
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(config["batch_size"])

    return dataset


def val_data_creator(config):
    _, (val_feature, val_label) = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((val_feature, val_label))
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(config["batch_size"])

    return dataset


def model_creator(config):
    model = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                input_shape=(28, 28, 1), padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(500, activation='tanh'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main(max_epoch):

    batch_size = 320
    config = {
        "batch_size": batch_size
    }
    est = Estimator.from_keras(model_creator, config=config, workers_per_node=2)
    stats = est.fit(train_data_creator,
                    epochs=max_epoch,
                    steps_per_epoch=60000 // batch_size,
                    validation_data_creator=val_data_creator,
                    validation_steps=10000 // batch_size)
    print(stats)
    est.save("/tmp/mnist_keras.ckpt")
    est.restore("/tmp/mnist_keras.ckpt")
    stats = est.evaluate(val_data_creator, steps=10000 // batch_size)
    print(stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')
    parser.add_argument('--max_epoch', type=int, default=5,
                        help='max epoch')

    args = parser.parse_args()
    if args.cluster_mode == "local":
        init_orca_context(cluster_mode="local", cores=4, init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2,
                          init_ray_on_spark=True, driver_memory="6g")
    main(args.max_epoch)

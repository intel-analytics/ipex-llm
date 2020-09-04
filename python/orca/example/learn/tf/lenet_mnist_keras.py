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
from zoo.orca.learn.tf.estimator import Estimator


def preprocess(x, y):
    return tf.to_float(tf.reshape(x, (-1, 28, 28, 1))) / 255.0, y


def main(max_epoch):

    # get DataSet
    (train_feature, train_label), (val_feature, val_label) = tf.keras.datasets.mnist.load_data()

    # tf.data.Dataset.from_tensor_slices is for demo only. For production use, please use
    # file-based approach (e.g. tfrecord).
    train_dataset = tf.data.Dataset.from_tensor_slices((train_feature, train_label))
    train_dataset = train_dataset.map(preprocess)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_feature, val_label))
    val_dataset = val_dataset.map(preprocess)

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

    est = Estimator.from_keras(keras_model=model)
    est.fit(data=train_dataset,
            batch_size=320,
            epochs=max_epoch,
            validation_data=val_dataset)

    result = est.evaluate(val_dataset)
    print(result)

    est.save_keras_model("/tmp/mnist_keras.h5")

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

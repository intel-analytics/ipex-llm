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
# ==============================================================================
import os
import pathlib

import tensorflow as tf
from tensorflow.keras import layers
from bigdl.nano.tf.keras import Sequential

if "FTP_URI" in os.environ:
    URI = os.environ['FTP_URI']
    dataset_url = URI + "/BigDL-data/flower_photos.tar.gz"
else:
    dataset_url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
batch_size, img_height, img_width = 32, 180, 180


def dataset_generation():
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    return num_classes, train_ds, val_ds


def model_init(num_classes):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def test_fit_function():
    num_classes, train_ds, val_ds = dataset_generation()

    # Case 1: Default
    model_default = model_init(num_classes)
    history_default = model_default.fit(train_ds, epochs=3, validation_data=val_ds)

    # Case 2: Multiple processing argument
    # Case 2.1: multiple processing backend
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, validation_data=val_ds,
                                                  num_processes=2, backend="multiprocessing")
    assert (history_default.history['accuracy'][-1]
            - history_multiprocess.history['accuracy'][-1]) <= 0.1


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

import os
import pathlib

import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from bigdl.nano.tf.keras import Sequential

# URI = os.environ['FTP_URI']

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
batch_size, img_height, img_width = 32, 180, 180
results_dir = os.path.join(os.path.dirname(__file__), "../results")


def write_result_to_csv(result, columns, result_path):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    df = pd.DataFrame(
        result,
        columns=columns
    )
    df.to_csv(result_path, index=False, sep=',')


def dataset_generation():
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


def main():
    num_classes, train_ds, val_ds = dataset_generation()

    model_default = model_init(num_classes)
    his_default = model_default.fit(train_ds, epochs=3, validation_data=val_ds)

    model_multiprocess = model_init(num_classes)
    his_multiprocess = model_multiprocess.fit(train_ds, epochs=3, validation_data=val_ds,
                           nprocs=2, backend="multiprocessing")

    write_result_to_csv(
        list(zip(his_default.history["loss"], his_multiprocess.history["loss"])),
        ["loss_default", "loss_multiprocess"],
        os.path.join(results_dir, "keras_multiporcessing_loss.csv")
    )

    write_result_to_csv(
        list(zip(his_default.history["accuracy"], his_multiprocess.history["accuracy"])),
        ["accuracy_default", "accuracy_multiprocess"],
        os.path.join(results_dir, "keras_multiporcessing_accuracy.csv")
    )


if __name__ == '__main__':
    main()

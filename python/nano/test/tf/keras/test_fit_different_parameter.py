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
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from bigdl.nano.tf.keras import Sequential

BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH = 32, 180, 180


def dataset_generation():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    return num_classes, train_ds, val_ds


def model_init(num_classes):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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


def generate_data_label():
    num_classes = 3
    train_examples = np.random.random((96, IMG_HEIGHT, IMG_WIDTH, 3))
    train_labels = np.random.randint(0, num_classes, size=(96,))
    return num_classes, train_examples, train_labels


def test_default():
    num_classes, train_ds, val_ds = dataset_generation()

    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3,
                                                  validation_data=val_ds, nprocs=2)


def test_batch_size():
    num_classes, train_ds, val_ds = dataset_generation()

    train_ds.unbatch()
    val_ds.unbatch()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, batch_size=32,
                                                  validation_data=val_ds, nprocs=2,
                                                  validation_batch_size=32)


def test_verbose():
    num_classes, train_ds, val_ds = dataset_generation()

    # verbose = 0
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, verbose=0,
                                                  validation_data=val_ds, nprocs=2)

    # verbose = 1
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, verbose=1,
                                                  validation_data=val_ds, nprocs=2)

    # verbose = 2
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, verbose=2,
                                                  validation_data=val_ds, nprocs=2)


def test_callbacks():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('loss') < 0.4:
                self.model.stop_training = True

    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, validation_data=val_ds,
                                                  callbacks=myCallback(), nprocs=2)


def test_validation_split():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3,
                                                  validation_split=0.2, nprocs=2)


def test_shuffle():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, nprocs=2, shuffle=True)


def test_class_weight():
    # just support tf.data.Dataset for now
    num_classes, train_data, train_label = generate_data_label()
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_label),
                                                      y=train_label)
    class_weights = {i: class_weights[i] for i in range(num_classes)}
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_data, train_label, epochs=3,
                                                  class_weight=class_weights, nprocs=2)


def test_sample_weight():
    # just support tf.data.Dataset for now
    num_classes, train_data, train_label = generate_data_label()
    from sklearn.utils import class_weight
    sample_weights = class_weight.compute_sample_weight(class_weight='balanced',
                                                        y=train_label)
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_data, train_label, epochs=3,
                                                  sample_weight=sample_weights, nprocs=2)


def test_initial_epoch():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, initial_epoch=1,
                                                  validation_data=val_ds, nprocs=2)


def test_steps_per_epoch():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, steps_per_epoch=50,
                                                  validation_data=val_ds, nprocs=2)


def test_validation_steps():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, validation_data=val_ds,
                                                  validation_steps=10, nprocs=2)


def test_validation_freq():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=5, validation_data=val_ds,
                                                  validation_freq=2, nprocs=2)


def test_max_queue_size():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, validation_data=val_ds,
                                                  max_queue_size=50, nprocs=2)


def test_workers():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, validation_data=val_ds,
                                                  workers=10, nprocs=2)


def test_use_multiprocessing():
    num_classes, train_ds, val_ds = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=3, validation_data=val_ds,
                                                  use_multiprocessing=True, nprocs=2)

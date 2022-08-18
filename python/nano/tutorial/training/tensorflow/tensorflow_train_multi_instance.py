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


import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import EfficientNetB0
import tensorflow_datasets as tfds

# Use `Model` and `Sequential` in `bigdl.nano.tf.keras` instead of tensorflow's
from bigdl.nano.tf.keras import Model


def create_datasets(img_size, batch_size):
    (ds_train, ds_test), ds_info = tfds.load(
        "stanford_dogs",
        split=['train', 'test'],
        with_info=True,
        as_supervised=True
    )

    num_classes = ds_info.features['label'].num_classes

    data_augmentation = Sequential([
        layers.RandomFlip(),
        layers.RandomRotation(factor=0.15),
    ])

    def preprocessing(img, label):
        img, label =  tf.image.resize(img, (img_size, img_size)), tf.one_hot(label, num_classes)
        return data_augmentation(img), label

    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.cache().repeat().map(preprocessing). \
        batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    ds_test = ds_test.map(preprocessing). \
        batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    return ds_train, ds_test, ds_info


def create_model(num_classes, img_size, learning_rate=1e-2):
    inputs = layers.Input(shape = (img_size, img_size, 3))

    backbone = EfficientNetB0(include_top=False, input_tensor=inputs)

    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name='avg_pool')(backbone.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = Model(inputs, outputs, name='EfficientNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    img_size = 224
    batch_size = 32
    num_epochs = 1
    
    ds_train, ds_test, ds_info = create_datasets(img_size=img_size, batch_size=batch_size)
    
    num_classes = ds_info.features['label'].num_classes
    steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
    validation_steps = ds_info.splits['test'].num_examples // batch_size

    # Multi-Instance Training
    # 
    # It is often beneficial to use multiple instances for training
    # if a server contains multiple sockets or many cores, 
    # so that the workload can make full use of all CPU cores.
    # BigDL-Nano makes it very easy to conduct multi-instance training correctly.
    # 
    # Use `Model` or `Sequential` in `bigdl.nano.tf.keras` to create model,
    # then just set the `num_processes` parameter in the `fit` method.
    # BigDL-Nano will launch the specific number of processes to perform data-parallel training.
    #
    model = create_model(num_classes=num_classes, img_size=img_size)
    model.fit(ds_train,
              epochs=num_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=ds_test,
              validation_steps=validation_steps,
              num_processes=4)

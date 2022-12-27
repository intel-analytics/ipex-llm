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

# This example shows how to do multi-process training with bigdl-nano


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import tensorflow_datasets as tfds

# Use `Model` and `Sequential` in `bigdl.nano.tf.keras` instead of tensorflow's
# Use `InferenceOptimizer` for OpenVINO acceleration
from bigdl.nano.tf.keras import InferenceOptimizer


def create_datasets(img_size, batch_size):
    (train_ds, test_ds), info = tfds.load('imagenette/320px-v2',
                                          data_dir='/tmp/data',
                                          split=['train', 'validation'],
                                          with_info=True,
                                          as_supervised=True)
    
    num_classes = info.features['label'].num_classes
    
    def preprocessing(img, label):
        return tf.image.resize(img, (img_size, img_size)), \
               tf.one_hot(label, num_classes)

    train_ds = train_ds.repeat().map(preprocessing).batch(batch_size)
    test_ds = test_ds.map(preprocessing).batch(batch_size)
    return train_ds, test_ds, info


def create_model(num_classes, img_size):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    backbone = ResNet50(weights='imagenet')
    backbone.trainable = False
    x = backbone(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


if __name__ == '__main__':
    img_size = 224
    batch_size = 32
    num_epochs = int(os.environ.get('NUM_EPOCHS', 10))

    train_ds, test_ds, ds_info = create_datasets(img_size, batch_size)
    num_classes = ds_info.features['label'].num_classes
    steps_per_epoch = ds_info.splits['train'].num_examples // batch_size

    model = create_model(num_classes, img_size)

    # trace the tensorflow.keras model to an openvino model
    openvino_model = InferenceOptimizer.trace(model, accelerator="openvino")

    # use the traced model same as the origial model
    data_example = np.random.random((1, 224, 224, 3))
    y = openvino_model(data_example)
    print(y)

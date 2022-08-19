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

# Required Dependecies

# ```bash
# pip install neural-compressor==1.11.0
# ```


import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow_datasets as tfds

# Use `Model` and `Sequential` in `bigdl.nano.tf.keras` instead of tensorflow's
from bigdl.nano.tf.keras import Model, Sequential


def create_datasets(img_size, batch_size):
    (train_ds, test_ds), info = tfds.load('imagenette/320px-v2',
                                          data_dir='/tmp/data',
                                          split=['train', 'validation'],
                                          with_info=True,
                                          as_supervised=True)

    # Create a Dataset that includes only 1/num_shards of full dataset.
    num_shards = int(os.environ.get('NUM_SHARDS', 1))
    train_ds = train_ds.shard(num_shards, index=0)
    test_ds = test_ds.shard(num_shards, index=0)
    num_classes = info.features['label'].num_classes
    
    train_ds = train_ds.map(lambda img, label: (tf.image.resize(img, (img_size, img_size)),
                                                tf.one_hot(label, num_classes))).batch(batch_size)
    test_ds = test_ds.map(lambda img, label: (tf.image.resize(img, (img_size, img_size)),
                                              tf.one_hot(label, num_classes))).batch(batch_size)
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

    train_ds, test_ds, ds_info = create_datasets(img_size, batch_size)
    num_classes = ds_info.features['label'].num_classes

    # Quantization with Intel Neural Compressor
    #
    # A quantized model executes tensor operations with reduced precision.
    # This allows for a more compact model representation and the use of high 
    # performance vectorized operations on many hardware platforms.
    # It can be used to speed up inference and only the forward pass is supported
    # for quantized operators.
    # 
    # Use `Model` or `Sequential` in `bigdl.nano.tf.keras` to create a model, then
    # `Model.quantize()` return a Keras module with desired precision and accuracy.
    # Taking Resnet50 as an example, you can add quantization as below.
    #
    model = create_model(num_classes, img_size)
    q_model = model.quantize(calib_dataset=test_ds,
                             metric=CategoricalAccuracy(),
                             tuning_strategy='basic')

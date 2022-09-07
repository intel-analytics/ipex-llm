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
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow_datasets as tfds

# Use `Model` and `Sequential` in `bigdl.nano.tf.keras` instead of tensorflow's
from bigdl.nano.tf.keras import Model


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

    # Multi-Instance Training
    # 
    # It is often beneficial to use multiple instances for training
    # if a server contains multiple sockets or many cores, 
    # so that the workload can make full use of all CPU cores.
    #
    # When using data-parallel training, the batch size is equivalent to
    # becoming n times larger, where n is the number of parallel processes.
    # We should to scale the learning rate to n times as well to achieve the
    # same effect as single instance training.
    # However, scaling the learning rate linearly may lead to poor convergence
    # at the beginning of training, so we should gradually increase the
    # learning rate to n times, and this is called 'learning rate warmup'.
    # 
    # Fortunately, BigDL-Nano makes it very easy to conduct multi-instance 
    # training correctly. It will handle all these for you.
    # 
    # Use `Model` or `Sequential` in `bigdl.nano.tf.keras` to create model,
    # then just set the `num_processes` parameter in the `fit` method.
    # BigDL-Nano will launch the specific number of processes to perform
    # data-parallel training, in addition, it will automatically apply
    # learning rate scaling and warmup for your training.
    #
    model = create_model(num_classes, img_size)
    model.fit(train_ds,
              epochs=num_epochs,
              steps_per_epoch=steps_per_epoch,
              num_processes=2)

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
# ==============================================================================
# Most of the code is adapted from
# https://keras.io/guides/transfer_learning/
# #an-endtoend-example-finetuning-an-image-classification-model
# -on-a-cats-vs-dogs-dataset
#

# Step 0: Import necessary libraries
import math

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator

# Step 1: Init Orca Context
init_orca_context(cluster_mode="local")


# Step 2: Read and process data
def data_process():
    train_ds, validation_ds, test_ds = tfds.load(
        "cats_vs_dogs",
        # Reserve 10% for validation and 10% for test
        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
        as_supervised=True,  # Include labels
    )

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
    print(
        "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
    )
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

    return train_ds, validation_ds, test_ds


# TODO: remove this if train/val steps are not required
train_ds, validation_ds, test_ds = data_process()


def preprocess(x, y):
    x = tf.image.resize(x, (150, 150))
    return x, y


def train_data_creator(config, batch_size):
    dataset = tfds.load("cats_vs_dogs",
                        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
                        as_supervised=True)[0]
    dataset = dataset.map(preprocess).batch(batch_size)
    return dataset


def val_data_creator(config, batch_size):
    dataset = tfds.load("cats_vs_dogs",
                        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
                        as_supervised=True)[1]
    dataset = dataset.map(preprocess).batch(batch_size)
    return dataset


def test_data_creator(config, batch_size):
    dataset = tfds.load("cats_vs_dogs",
                        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
                        as_supervised=True)[2]
    dataset = dataset.map(preprocess).batch(batch_size)
    return dataset


# Step 3: Define model
def xception_model(dropout):
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False
    data_augmentation = keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), ]
    )
    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    return model


def model_creator(config):
    model = xception_model(config['dropout'])
    return model


# Step 4: Distributed training with Orca keras Estimator
backend = 'spark'  # 'ray' or 'spark'
est = Estimator.from_keras(model_creator=model_creator,
                           config={"dropout": 0.2},
                           backend=backend)

batch_size = 32
train_steps = math.ceil(tf.data.experimental.cardinality(train_ds) / batch_size)
val_steps = math.ceil(tf.data.experimental.cardinality(validation_ds) / batch_size)
test_steps = math.ceil(tf.data.experimental.cardinality(test_ds) / batch_size)
train_stats = est.fit(data=train_data_creator,
                      epochs=1,
                      batch_size=batch_size,
                      steps_per_epoch=train_steps,
                      validation_data=val_data_creator,
                      validation_steps=val_steps)
print("Train results:")
for k, v in train_stats.items():
    print("{}: {}".format(k, v))


# Step 5: Distributed evaluation of the trained model
eval_stats = est.evaluate(test_data_creator,
                          batch_size=batch_size,
                          num_steps=test_steps)
print("Evaluation results:")
for k, v in eval_stats.items():
    print("{}: {}".format(k, v))


# Step 6: Save the trained Tensorflow model
est.save("xception_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()

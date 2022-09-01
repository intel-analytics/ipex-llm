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

# This file is adapted from Keras Tutorial.
# https://github.com/keras-team/keras-io/blob/
# master/examples/vision/mnist_convnet.py

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import losses, metrics

from bigdl.nano.tf.keras import Model

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model = Model(inputs=model.inputs, outputs=model.outputs)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# Quantization using Intel Neural Compressor
# pip install 'neural-compressor'

# calib_dataset only accept tf.data.Dataset object
tune_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Execute quantization
q_model = model.quantize(precision='int8',
                         accelerator=None,
                         calib_dataset=tune_dataset,
                         accuracy_criterion={
                             'relative': 0.01, 'higher_is_better': True},
                         approach='static',
                         tuning_strategy='bayesian',
                         timeout=0,
                         max_trials=10,
                         )

y_test_hat = q_model(x_test)
loss = float(tf.reduce_mean(
             losses.categorical_crossentropy(y_test, y_test_hat)))
categorical_accuracy = metrics.CategoricalAccuracy()
categorical_accuracy.update_state(y_test, y_test_hat)
accuracy = categorical_accuracy.result().numpy()
print("Quantization test loss:", loss)
print("Quantization test accuracy:", accuracy)

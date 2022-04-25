
import bigdl.nano.automl as automl
automl.hpo_config.enable_hpo_tf()

import tensorflow as tf
from tensorflow import keras
from bigdl.nano.tf.keras.layers import Dense
from bigdl.nano.tf.keras import Input
from bigdl.nano.automl.tf.keras import Model
import bigdl.nano.automl.hpo.space as space
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

from time import time

inputs = Input(shape=(784,))

# Just for demonstration purposes.
img_inputs = Input(shape=(32, 32, 3))

x = Dense(units=space.Categorical(8,16), activation="relu")(inputs)
x = Dense(units=space.Categorical(32,64), activation="relu")(x)
outputs = Dense(units=10)(x)


model = Model(inputs=inputs, outputs=outputs, name="mnist_model")


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)


t1 = time()

model.search(
    n_trials=2, #
    target_metric='accuracy',
    direction="maximize",
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=2,
    validation_split=0.2,
    verbose=False,
)

t2 = time()

print("search took", (t2-t1), "secs")

history = model.fit(x_train, y_train,
                    batch_size=128, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

print(model.summary())

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


import pytest
from unittest import TestCase

import bigdl.nano.automl as automl
automl.hpo_config.enable_hpo_tf()

import numpy as np
import bigdl.nano.automl.hpo.space as space
from bigdl.nano.tf.keras import Input
from bigdl.nano.tf.keras.layers import Dense
from bigdl.nano.automl.tf.keras import Model

from tensorflow.keras.optimizers import Adam

class TestFunctional(TestCase):


    def test_functional_fit(self):
        train_samples = 1000
        val_samples = 100
        dim = 98
        np.random.seed(1337)
        x_train = np.random.rand(train_samples, dim)
        x_test = np.random.randn(val_samples, dim)
        y_train = np.random.rand(train_samples, dim)
        y_test = np.random.randn(val_samples, dim)

        inputs = Input(shape=(dim,))
        x = Dense(units=space.Categorical(8,16), activation="linear")(inputs)
        x = Dense(units=space.Categorical(32,64), activation="tanh")(x)
        outputs = Dense(units=dim)(x)

        model = Model(inputs=inputs, outputs=outputs, name="mnist_model")

        model.compile(loss="mean_squared_error",
                      optimizer=Adam(learning_rate=0.0001),
                      metrics=["accuracy"])

        model.search(n_trials=2,target_metric='accuracy', direction="maximize",
            x=x_train, y=y_train,
            batch_size=128, epochs=2, validation_split=0.2)
        history = model.fit(x_train, y_train,
                    batch_size=128, epochs=2, validation_split=0.2)

        test_scores = model.evaluate(x_test, y_test, verbose=2)



if __name__ == '__main__':
    pytest.main([__file__])
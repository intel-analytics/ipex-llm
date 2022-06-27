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

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import bigdl.nano.automl.hpo.space as space
from bigdl.nano.tf.keras import Input
from bigdl.nano.tf.keras.layers import Dense, Conv2D, Flatten, Dropout
from bigdl.nano.automl.tf.keras import Model, Sequential

from bigdl.nano.tf.optimizers import Adam, RMSprop
import bigdl.nano.automl.hpo as hpo
from bigdl.nano.automl.hpo.backend import PrunerType, SamplerType

@hpo.tfmodel()
class MyModel(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, num_classes=10):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            activation="relu")
        self.max1  = tf.keras.layers.MaxPooling2D(3)
        self.bn1   = tf.keras.layers.BatchNormalization()

        self.gap   = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.bn1(x)
        x = self.gap(x)
        return self.dense(x)


class TestUseCases(TestCase):

    def __init__(self,*args, **kwargs):
        super(TestUseCases, self).__init__(*args, **kwargs)
        self.TRAIN_TOTAL_SAMPLES = 3000 # use less data
        self.VALID_TOTAL_SAMPLES = 1000 # use less data
        self.TEST_TOTAL_SAMPLES = 1000
        np.random.seed(1337)
        self._create_data()

    def _create_data(self):
        # (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
        self.x_train = np.random.randn(self.TRAIN_TOTAL_SAMPLES,28,28)
        self.x_valid = np.random.randn(self.VALID_TOTAL_SAMPLES,28,28)
        self.x_test = np.random.randn(self.TEST_TOTAL_SAMPLES,28,28)
        self.y_train = np.random.randint(10, size=(self.TRAIN_TOTAL_SAMPLES,))
        self.y_valid = np.random.randint(10, size=(self.VALID_TOTAL_SAMPLES,))
        self.y_test = np.random.randint(10, size=(self.TEST_TOTAL_SAMPLES,))

    def test_functional(self):
        n_samples_train=self.TRAIN_TOTAL_SAMPLES
        n_samples_test=self.TEST_TOTAL_SAMPLES
        # prepare data
        x_train = self.x_train[:n_samples_train].reshape(n_samples_train, 784)
        x_test = self.x_test[:n_samples_test].reshape(n_samples_test, 784)
        y_train = self.y_train[:n_samples_train]
        y_test = self.y_test[:n_samples_test]

        #define the model
        inputs = Input(shape=(784,))
        x = Dense(units=space.Categorical(8,16,prefix='dense_1'), activation="linear")(inputs)
        x = Dense(units=space.Categorical(32,64,prefix='dense_2'), activation="tanh")(x)
        x = Dropout(rate=space.Real(0.1,0.5, prefix='dropout'))(x)
        outputs = Dense(units=10)(x)
        model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=RMSprop(
                learning_rate=space.Real(0.0001, 0.01, log=True)),
            metrics=["accuracy"],
        )
        # run hpo
        model.search(n_trials=2,
                     target_metric='accuracy',
                     direction="maximize",
                     sampler=SamplerType.Random,
                     pruner=PrunerType.HyperBand,
                     pruner_kwargs={'min_resource':1, 'max_resource':100, 'reduction_factor':3},
                     x=x_train,
                     y=y_train,
                     batch_size=space.Categorical(128,64),
                     epochs=2,
                     validation_split=0.2)
        study = model.search_summary()
        assert(study.best_trial)
        assert('dense_1:units'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('dense_2:units'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('dropout:rate' in study.best_trial.params)
        assert('learning_rate' in study.best_trial.params)
        assert('batch_size'+space.SPLITTER+'choice' in study.best_trial.params)
        # run fit
        history = model.fit(x_train, y_train,
                    batch_size=128, epochs=2, validation_split=0.2)
        test_scores = model.evaluate(x_test, y_test, verbose=2)

    def test_sequential(self):
        n_samples_train=self.TRAIN_TOTAL_SAMPLES
        n_samples_valid=self.TEST_TOTAL_SAMPLES
        # prepare data
        img_x, img_y = self.x_train.shape[1], self.x_train.shape[2]
        x_train = self.x_train.reshape(-1, img_x, img_y,1)[:n_samples_train]
        x_valid = self.x_valid.reshape(-1, img_x, img_y,1)[:n_samples_valid]
        y_train = self.y_train[:n_samples_train]
        y_valid = self.y_valid[:n_samples_valid]
        input_shape = (img_x, img_y, 1)
        # define the model as usual.
        model = Sequential()
        model.add(Conv2D(
            filters=space.Categorical(32, 64),
            kernel_size=space.Categorical(3, 5),
            strides=space.Categorical(1, 2),
            activation=space.Categorical("relu", "linear"),
            input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=0.0001),
            metrics=["accuracy"]
        )
        # run hpo
        model.search(
            n_trials=2,
            target_metric='accuracy',
            direction="maximize",
            x=x_train,
            y=y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            # batch_size=128,
            batch_size=space.Int(128,256, prefix='global'),
            epochs=2
        )
        study = model.search_summary()
        assert(study.best_trial)
        assert('filters'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('kernel_size'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('strides'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('global:batch_size' in study.best_trial.params)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            batch_size=128,
            epochs=2,
            verbose=False
        )

        # print(model.summary())
        score = model.evaluate(x_valid, y_valid, verbose=0)

    def test_custom(self):
        n_samples_train=self.TRAIN_TOTAL_SAMPLES
        n_samples_valid=self.TEST_TOTAL_SAMPLES
        # prepare data
        img_x, img_y = self.x_train.shape[1], self.x_train.shape[2]
        x_train = self.x_train.reshape(-1, img_x, img_y,1)[:n_samples_train]
        x_valid = self.x_valid.reshape(-1, img_x, img_y,1)[:n_samples_valid]
        y_train = self.y_train[:n_samples_train]
        y_valid = self.y_valid[:n_samples_valid]
        input_shape = (img_x, img_y, 1)
        # set search spaces
        model = MyModel(
            filters=hpo.space.Categorical(32, 64),
            kernel_size=hpo.space.Categorical(3, 5),
            strides=hpo.space.Categorical(1, 2)
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=RMSprop(learning_rate=1e-4),
            metrics=["accuracy"],
        )
        # run hpo
        model.search(
            n_trials=2,
            target_metric='accuracy',
            direction="maximize",
            target_metric_mode='auto',
            x=x_train,
            y=y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            batch_size=space.Categorical(128,64, prefix='fit'),
            epochs=2,
            verbose=False,
        )
        study = model.search_summary()
        assert(study.best_trial)
        assert('filters'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('kernel_size'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('strides'+space.SPLITTER+'choice' in study.best_trial.params)
        assert('fit:batch_size'+space.SPLITTER+'choice' in study.best_trial.params)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            batch_size=120,
            epochs=2,
            verbose=True,
        )
        # print(model.summary())
        score = model.evaluate(x_valid, y_valid, verbose=0)
        # print("The final score is on test data is", score[1])

    def test_fit_without_search_with_space(self):
        n_samples_train=self.TRAIN_TOTAL_SAMPLES
        n_samples_test=self.TEST_TOTAL_SAMPLES
        # prepare data
        x_train = self.x_train[:n_samples_train].reshape(n_samples_train, 784)
        x_test = self.x_test[:n_samples_test].reshape(n_samples_test, 784)
        y_train = self.y_train[:n_samples_train]
        y_test = self.y_test[:n_samples_test]

        #define the model
        inputs = Input(shape=(784,))
        x = Dense(units=space.Categorical(8,16), activation="linear")(inputs)
        x = Dense(units=space.Categorical(32,64), activation="tanh")(x)
        outputs = Dense(units=10)(x)
        model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )

        # run fit
        with self.assertRaises(ValueError):
            history = model.fit(x_train, y_train,
                    batch_size=128, epochs=2, validation_split=0.2)


    def test_fit_without_search_without_space(self):
        n_samples_train=self.TRAIN_TOTAL_SAMPLES
        n_samples_test=self.TEST_TOTAL_SAMPLES
        # prepare data
        x_train = self.x_train[:n_samples_train].reshape(n_samples_train, 784)
        x_test = self.x_test[:n_samples_test].reshape(n_samples_test, 784)
        y_train = self.y_train[:n_samples_train]
        y_test = self.y_test[:n_samples_test]

        #define the model
        inputs = Input(shape=(784,))
        x = Dense(units=8, activation="linear")(inputs)
        x = Dense(units=32, activation="tanh")(x)
        outputs = Dense(units=10)(x)
        model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )

        # run fit
        history = model.fit(x_train, y_train,
                    batch_size=128, epochs=2, validation_split=0.2)
        test_scores = model.evaluate(x_test, y_test, verbose=2)



if __name__ == '__main__':
    pytest.main([__file__])
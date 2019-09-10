#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.feature.common import ChainedPreprocessing, FeatureSet
from zoo.feature.image import *
from zoo.pipeline.api.net import TFOptimizer
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np
import os

from zoo.tfpark import KerasModel, TFDataset

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")


class TestTFParkModel(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        super(TestTFParkModel, self).setup_method(method)

    def create_model(self):
        data = tf.keras.layers.Input(shape=[10])

        x = tf.keras.layers.Flatten()(data)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=data, outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_multi_input_output_model(self):
        data1 = tf.keras.layers.Input(shape=[10])
        data2 = tf.keras.layers.Input(shape=[10])

        x1 = tf.keras.layers.Flatten()(data1)
        x1 = tf.keras.layers.Dense(10, activation='relu')(x1)
        pred1 = tf.keras.layers.Dense(2, activation='softmax')(x1)

        x2 = tf.keras.layers.Flatten()(data2)
        x2 = tf.keras.layers.Dense(10, activation='relu')(x2)
        pred2 = tf.keras.layers.Dense(2)(x2)

        model = tf.keras.models.Model(inputs=[data1, data2], outputs=[pred1, pred2])
        model.compile(optimizer='rmsprop',
                      loss=['sparse_categorical_crossentropy', 'mse'])
        return model

    def create_training_data(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))
        return x, y

    def create_training_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y)

        dataset = TFDataset.from_rdd(rdd,
                                     features=(tf.float32, [10]),
                                     labels=(tf.int32, []),
                                     batch_size=4,
                                     val_rdd=rdd
                                     )
        return dataset

    def create_evaluation_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y)

        dataset = TFDataset.from_rdd(rdd,
                                     features=(tf.float32, [10]),
                                     labels=(tf.int32, []),
                                     batch_per_thread=1
                                     )
        return dataset

    def create_predict_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)

        rdd = self.sc.parallelize(x)

        rdd = rdd.map(lambda x: [x])

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features"],
                                     shapes=[[10]],
                                     types=[tf.float32],
                                     batch_per_thread=1
                                     )
        return dataset

    def test_training_with_ndarray(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        model.fit(x, y, batch_size=2)

    def test_training_with_ndarry_distributed(self):
        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        model.fit(x, y, batch_size=4, distributed=True)

    def test_training_with_ndarry_distributed_twice(self):
        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        model.fit(x, y, batch_size=4, distributed=True)
        model.fit(x, y, batch_size=4, distributed=True)

    def test_training_with_validation_data(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        val_x, val_y = self.create_training_data()

        model.fit(x, y, validation_data=(val_x, val_y), batch_size=4)

    def test_training_with_validation_data_distributed(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        val_x, val_y = self.create_training_data()

        model.fit(x, y, validation_data=(val_x, val_y), batch_size=4, distributed=True)

    def test_training_and_validation_with_dataset(self):
        keras_model = self.create_model()
        model = KerasModel(keras_model)

        dataset = self.create_training_dataset()

        model.fit(dataset)

    def test_evaluate_with_ndarray(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        model.fit(x, y, batch_size=4, epochs=10)

        results_after = model.evaluate(x, y)

        assert results_pre[0] > results_after[0]

    def test_evaluate_with_ndarray_distributed(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        model.fit(x, y, batch_size=4, epochs=10)

        results_after = model.evaluate(x, y, distributed=True)

        assert results_pre[0] > results_after[0]

    def test_evaluate_and_distributed_evaluate(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        results_after = model.evaluate(x, y, distributed=True)

        assert np.square(results_pre[0] - results_after[0]) < 0.000001
        assert np.square(results_pre[1] - results_after[1]) < 0.000001

    def test_evaluate_with_dataset(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        dataset = self.create_evaluation_dataset()

        results_after = model.evaluate(dataset)

        assert np.square(results_pre[0] - results_after[0]) < 0.000001
        assert np.square(results_pre[1] - results_after[1]) < 0.000001

    def test_predict_with_ndarray(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        pred_y = np.argmax(model.predict(x), axis=1)

        acc = np.average((pred_y == y))

        assert np.square(acc - results_pre[1]) < 0.000001

    def test_predict_with_ndarray_distributed(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        pred_y = np.argmax(model.predict(x, distributed=True), axis=1)

        acc = np.average((pred_y == y))

        assert np.square(acc - results_pre[1]) < 0.000001

    def test_predict_with_dataset(self):

        keras_model = self.create_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        pred_y = np.argmax(np.array(model.predict(
            self.create_predict_dataset()).collect()), axis=1)

        acc = np.average((pred_y == y))

        assert np.square(acc - results_pre[1]) < 0.000001

    # move the test here to avoid keras session to be closed (not sure about why)
    def test_tf_optimizer_with_sparse_gradient_using_keras(self):
        import tensorflow as tf

        ids = np.random.randint(0, 10, size=[40])
        labels = np.random.randint(0, 5, size=[40])
        id_rdd = self.sc.parallelize(ids)
        label_rdd = self.sc.parallelize(labels)
        training_rdd = id_rdd.zip(label_rdd).map(lambda x: [x[0], x[1]])

        dataset = TFDataset.from_rdd(training_rdd,
                                     names=["ids", "labels"],
                                     shapes=[[], []],
                                     types=[tf.int32, tf.int32],
                                     batch_size=8)
        words_input = tf.keras.layers.Input(shape=(), name='words_input')
        embedding_layer = tf.keras.layers.Embedding(input_dim=10,
                                                    output_dim=5, name='word_embedding')
        word_embeddings = embedding_layer(words_input)
        embedding = tf.keras.layers.Flatten()(word_embeddings)
        output = tf.keras.layers.Dense(5, activation="softmax")(embedding)
        model = tf.keras.models.Model(inputs=[words_input], outputs=[output])
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")

        optimizer = TFOptimizer.from_keras(model, dataset)
        optimizer.optimize()

    def test_tensorflow_optimizer(self):
        data = tf.keras.layers.Input(shape=[10])

        x = tf.keras.layers.Flatten()(data)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=data, outputs=predictions)
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        keras_model = KerasModel(model)

        x, y = self.create_training_data()

        keras_model.fit(x, y, batch_size=4, distributed=True)

if __name__ == "__main__":
    pytest.main([__file__])

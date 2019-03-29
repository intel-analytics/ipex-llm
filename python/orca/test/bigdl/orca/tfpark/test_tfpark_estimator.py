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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np

from zoo.tfpark import TFDataset
from zoo.tfpark.estimator import TFEstimatorSpec, TFEstimator


class TestTFParkEstimator(ZooTestCase):

    def create_model_fn(self):
        def model_fn(features, labels, mode):

            h1 = tf.layers.dense(features, 64, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
            logits = tf.layers.dense(h2, 10)

            if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
                loss = tf.reduce_mean(
                    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
                return TFEstimatorSpec(mode, predictions=logits, loss=loss)
            else:
                return TFEstimatorSpec(mode, predictions=logits)
        return model_fn

    def create_input_fn(self):

        def input_fn(mode):
            np.random.seed(20)
            x = np.random.rand(20, 10)
            y = np.random.randint(0, 10, (20))

            rdd_x = self.sc.parallelize(x)
            rdd_y = self.sc.parallelize(y)

            rdd = rdd_x.zip(rdd_y)
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                dataset = TFDataset.from_rdd(rdd,
                                             features=(tf.float32, [10]),
                                             labels=(tf.int32, []),
                                             batch_size=4)
            elif mode == tf.estimator.ModeKeys.EVAL:
                dataset = TFDataset.from_rdd(rdd,
                                             features=(tf.float32, [10]),
                                             labels=(tf.int32, []),
                                             batch_per_thread=4)
            else:
                dataset = TFDataset.from_rdd(rdd_x,
                                             features=(tf.float32, [10]),
                                             batch_per_thread=4)
            return dataset

        return input_fn

    def test_init_TFDataset_from_ndarrays(self):

        model_fn = self.create_model_fn()

        def input_fn(mode):
            x = np.random.rand(20, 10)
            y = np.random.randint(0, 10, (20,))
            if mode == tf.estimator.ModeKeys.TRAIN:
                return TFDataset.from_ndarrays((x, y), batch_size=8)
            elif mode == tf.estimator.ModeKeys.EVAL:
                return TFDataset.from_ndarrays((x, y), batch_per_thread=1)
            else:
                return TFDataset.from_ndarrays(x, batch_per_thread=1)

        estimator = TFEstimator(model_fn, tf.train.AdamOptimizer())
        estimator.train(input_fn, 10)
        estimator.evaluate(input_fn, ["acc"])
        estimator.predict(input_fn)

    def test_training(self):
        model_fn = self.create_model_fn()
        input_fn = self.create_input_fn()
        estimator = TFEstimator(model_fn, tf.train.AdamOptimizer())
        estimator.train(input_fn, steps=60000 // 320)

    def test_evaluating(self):
        model_fn = self.create_model_fn()
        input_fn = self.create_input_fn()
        estimator = TFEstimator(model_fn, tf.train.AdamOptimizer())
        eval_results = estimator.evaluate(input_fn, ["acc"])
        assert len(eval_results) > 0

    def test_predict(self):
        model_fn = self.create_model_fn()
        input_fn = self.create_input_fn()
        estimator = TFEstimator(model_fn, tf.train.AdamOptimizer())
        results = estimator.predict(input_fn).collect()

    def test_estimator_without_batch(self):
        def model_fn(features, labels, mode):

            assert features.shape.ndims == 1
            if labels is not None:
                assert labels.shape.ndims == 0

            features = tf.expand_dims(features, axis=0)

            h1 = tf.layers.dense(features, 64, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
            logits = tf.layers.dense(h2, 10)

            if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
                labels = tf.expand_dims(labels, axis=0)
                loss = tf.reduce_mean(
                    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
                return TFEstimatorSpec(mode, predictions=logits, loss=loss)
            else:
                return TFEstimatorSpec(mode, predictions=logits)

        def input_fn(mode):
            np.random.seed(20)
            x = np.random.rand(20, 10)
            y = np.random.randint(0, 10, (20))

            rdd_x = self.sc.parallelize(x)
            rdd_y = self.sc.parallelize(y)

            rdd = rdd_x.zip(rdd_y)
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                dataset = TFDataset.from_rdd(rdd,
                                             features=(tf.float32, [10]),
                                             labels=(tf.int32, []))
            else:
                dataset = TFDataset.from_rdd(rdd_x,
                                             features=(tf.float32, [10]))
            return dataset

        estimator = TFEstimator(model_fn, tf.train.AdamOptimizer())

        self.intercept(lambda: estimator.train(input_fn, steps=1),
                       "The batch_size of TFDataset must be specified when used for training.")

        estimator.evaluate(input_fn, ["acc"])
        estimator.predict(input_fn).collect()


if __name__ == "__main__":
    pytest.main([__file__])

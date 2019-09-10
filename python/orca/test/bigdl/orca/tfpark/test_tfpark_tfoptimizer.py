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

from bigdl.optim.optimizer import Adam, MaxEpoch
from zoo.pipeline.api.keras.metrics import Accuracy
from zoo.pipeline.api.net import TFOptimizer
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np
import os

from zoo.tfpark import KerasModel, TFDataset


class TestTFParkTFOptimizer(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        super(TestTFParkTFOptimizer, self).setup_method(method)

    def test_tf_optimizer_with_sparse_gradient(self):
        ids = np.random.randint(0, 10, size=[40])
        labels = np.random.randint(0, 5, size=[40])
        id_rdd = self.sc.parallelize(ids)
        label_rdd = self.sc.parallelize(labels)
        training_rdd = id_rdd.zip(label_rdd).map(lambda x: [x[0], x[1]])
        with tf.Graph().as_default():
            dataset = TFDataset.from_rdd(training_rdd,
                                         names=["ids", "labels"],
                                         shapes=[[], []],
                                         types=[tf.int32, tf.int32],
                                         batch_size=8)
            id_tensor, label_tensor = dataset.tensors
            embedding_table = tf.get_variable(
                name="word_embedding",
                shape=[10, 5])

            embedding = tf.nn.embedding_lookup(embedding_table, id_tensor)
            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=embedding,
                                                               labels=label_tensor))
            optimizer = TFOptimizer(loss, Adam(1e-3))
            optimizer.optimize(end_trigger=MaxEpoch(1))
            optimizer.sess.close()

    def test_tf_optimizer_metrics(self):

        features = np.random.randn(20, 10)
        labels = np.random.randint(0, 10, size=[20])
        with tf.Graph().as_default():
            dataset = TFDataset.from_ndarrays((features, labels),
                                              batch_size=4,
                                              val_tensors=(features, labels))
            feature_tensor, label_tensor = dataset.tensors
            output = tf.layers.dense(feature_tensor, 10)
            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=output,
                                                               labels=label_tensor))
            optimizer = TFOptimizer.from_loss(loss, Adam(1e-3),
                                              val_outputs=[output],
                                              val_labels=[label_tensor],
                                              val_method=Accuracy(), metrics={"loss": loss})
            optimizer.optimize(end_trigger=MaxEpoch(1))
            optimizer.sess.close()


if __name__ == "__main__":
    pytest.main([__file__])

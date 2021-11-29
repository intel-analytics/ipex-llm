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

from bigdl.dllib.optim.optimizer import Adam, SGD, MaxEpoch
from bigdl.dllib.keras.metrics import Accuracy
from bigdl.orca.test_zoo_utils import ZooTestCase
import tensorflow as tf
import numpy as np
import tempfile
import os

from bigdl.orca.tfpark import TFDataset, TFOptimizer


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
            optimizer = TFOptimizer.from_loss(loss, Adam(1e-3))
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
            features = tf.layers.dense(feature_tensor, 8)
            output = tf.layers.dense(features, 10)
            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=output,
                                                               labels=label_tensor))
            optimizer = TFOptimizer.from_loss(loss, {"dense/": Adam(1e-3), "dense_1/": SGD(0.0)},
                                              val_outputs=[output],
                                              val_labels=[label_tensor],
                                              val_method=Accuracy(), metrics={"loss": loss})
            initial_weights = optimizer.tf_model.training_helper_layer.get_weights()
            optimizer.optimize(end_trigger=MaxEpoch(1))
            updated_weights = optimizer.tf_model.training_helper_layer.get_weights()
            for i in [0, 1]:  # weights and bias combined with "dense/" should be updated
                assert not np.allclose(initial_weights[i], updated_weights[i])
            for i in [2, 3]:  # weights and bias combined with "dense_1" should be unchanged
                assert np.allclose(initial_weights[i], updated_weights[i])
            optimizer.sess.close()

    def test_control_inputs(self):

        features = np.random.randn(20, 10)
        labels = np.random.randint(0, 10, size=[20])
        with tf.Graph().as_default():
            dataset = TFDataset.from_ndarrays((features, labels),
                                              batch_size=4,
                                              val_tensors=(features, labels))
            is_training = tf.placeholder(dtype=tf.bool, shape=())
            feature_tensor, label_tensor = dataset.tensors
            features = tf.layers.dense(feature_tensor, 8)
            features = tf.layers.dropout(features, training=is_training)
            output = tf.layers.dense(features, 10)
            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=output,
                                                               labels=label_tensor))
            optimizer = TFOptimizer.from_loss(loss, Adam(),
                                              val_outputs=[output],
                                              val_labels=[label_tensor],
                                              val_method=Accuracy(),
                                              tensor_with_value={is_training: (True, False)},
                                              metrics={"loss": loss})
            optimizer.optimize(end_trigger=MaxEpoch(1))
            optimizer.sess.close()

    def test_checkpoint(self):

        features = np.random.randn(20, 10)
        labels = np.random.randint(0, 10, size=[20])
        with tf.Graph().as_default():
            dataset = TFDataset.from_ndarrays((features, labels),
                                              batch_size=4,
                                              val_tensors=(features, labels))
            feature_tensor, label_tensor = dataset.tensors
            features = tf.layers.dense(feature_tensor, 8)
            output = tf.layers.dense(features, 10)
            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=output,
                                                               labels=label_tensor))
            model_dir = tempfile.mkdtemp()
            try:
                optimizer = TFOptimizer.from_loss(loss, Adam(),
                                                  val_outputs=[output],
                                                  val_labels=[label_tensor],
                                                  val_method=Accuracy(),
                                                  metrics={"loss": loss}, model_dir=model_dir)
                optimizer.optimize(end_trigger=MaxEpoch(1))

                first_weights = optimizer.sess.run(tf.trainable_variables()[0])
                import re
                ckpt_path = None
                versions = []
                for (root, dirs, files) in os.walk(model_dir, topdown=True):
                    temp_versions = []
                    for file_name in files:
                        if re.match("^optimMethod-TFParkTraining\.[0-9]+$", file_name) is not None:
                            version = int(file_name.split(".")[1])
                            temp_versions.append(version)
                    if temp_versions:
                        ckpt_path = root
                        versions = temp_versions
                        break

                assert ckpt_path is not None, "Cannot fine checkpoint file"
                optimizer.sess.run(tf.global_variables_initializer())  # reset variable
                optimizer_load = TFOptimizer.from_loss(loss, Adam(),
                                                       session=optimizer.sess,
                                                       val_outputs=[output],
                                                       val_labels=[label_tensor],
                                                       val_method=Accuracy(),
                                                       metrics={"loss": loss}, model_dir=model_dir)
                optimizer_load.load_checkpoint(ckpt_path, max(versions))
                loaded_first_weights_before_train = optimizer.sess.run(tf.trainable_variables()[0])
                assert np.allclose(first_weights, loaded_first_weights_before_train)
                # max epoch still 1, should not train
                optimizer_load.optimize(end_trigger=MaxEpoch(1))
                loaded_first_weights = optimizer.sess.run(tf.trainable_variables()[0])
                assert np.allclose(first_weights, loaded_first_weights)

                # max epoch increase 1, should train 1 epoch
                optimizer_load.optimize(end_trigger=MaxEpoch(2))
                loaded_first_weights_2 = optimizer.sess.run(tf.trainable_variables()[0])
                assert not np.allclose(first_weights, loaded_first_weights_2)
                optimizer_load.sess.close()
            finally:
                import shutil
                shutil.rmtree(model_dir)


if __name__ == "__main__":
    pytest.main([__file__])

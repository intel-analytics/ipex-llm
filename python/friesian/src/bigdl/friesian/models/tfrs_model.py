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

import tensorflow as tf
import warnings
import tensorflow_recommenders as tfrs
from tensorflow_recommenders.tasks import base
from bigdl.dllib.utils import log4Error


class TFRSModel(tf.keras.Model):
    def __init__(self, tfrs_model):
        super().__init__()
        log4Error.invalidInputError(isinstance(tfrs_model, tfrs.Model),
                                    "FriesianTFRSModel only support tfrs.Model, but got " +
                                    tfrs_model.__class__.__name__)
        log4Error.invalidInputError(not tfrs_model._is_compiled,
                                    "TFRSModel should be initialized before compiling.")
        attr = tfrs_model.__dict__
        task_dict = dict()
        for k, v in attr.items():
            if isinstance(v, base.Task):
                task_dict[k] = v

        for k, v in task_dict.items():
            try:
                v._loss.reduction = tf.keras.losses.Reduction.NONE
            except:
                warnings.warn("Model task " + k + " has no attribute _loss, please use "
                                                  "`tf.keras.losses.Reduction.SUM` or "
                                                  "`tf.keras.losses.Reduction.NONE` for "
                                                  "loss reduction in this task if the "
                                                  "Estimator throw an error.")
        self.model = tfrs_model

    def call(self, features):
        return self.model.call(features)

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            loss = self.model.compute_loss(inputs, training=True)
            loss_rank = loss.shape.rank
            if loss_rank is not None and loss_rank != 0:
                loss = tf.nn.compute_average_loss(loss)

            # Handle regularization losses as well.
            regularization_loss = tf.cast(tf.nn.scale_regularization_loss(sum(self.model.losses)),
                                          tf.float32)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        # unscaled test loss
        loss = self.model.compute_loss(inputs, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.model.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

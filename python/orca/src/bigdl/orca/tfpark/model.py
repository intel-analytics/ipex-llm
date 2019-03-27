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

from bigdl.optim.optimizer import MaxEpoch
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras import models

from zoo.common.nncontext import getOrCreateSparkContext
from zoo.pipeline.api.net import TFDataset, TFOptimizer, TFPredictor
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class KerasModel(object):

    def __init__(self, model):
        self.model = model
        metrics_tensors = [
            self.model.metrics_tensors[m] for m in range(len(self.model.metrics_names) - 1)
        ]

        metrics_tensors = [self.model.total_loss] + metrics_tensors
        batch_size = tf.shape(model.inputs[0])

        def repeat(x, times):
            return tf.tile(tf.expand_dims(x, 0), tf.expand_dims(times, 0))

        self.metrics_tensors = [repeat(x, batch_size[0]) for x in metrics_tensors]

        self.tf_optimizer = None
        self.tf_optimizer_done_epochs = 0

    @property
    def metrics_names(self):
        return self.model.metrics_names

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save_weights(self, filepath, overwrite=True, save_format=None):
        self.model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name)

    def save_model(self, path):
        """
        Save the model to a single HDF5 file.

        :param path: String. The path to save the model.
        """
        self.model.save(path)

    @staticmethod
    def load_model(path):
        """
        Load an existing keras model (with weights) from HDF5 file.

        :param path: String. The path to the pre-defined model.
        :return: KerasModel.
        """
        return KerasModel(models.load_model(path))

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            validation_split=0.,
            validation_data=None,
            distributed=False,
            **kwargs
            ):
        if isinstance(x, TFDataset):
            # todo check arguments
            if not x.has_batch:
                raise ValueError("The batch_size of TFDataset must be " +
                                 "specified when used in KerasModel fit.")
            x = _standarize_feature_label_dataset(x, self.model)
            self._fit_distributed(x, validation_split, epochs, **kwargs)

        elif distributed:
            sc = getOrCreateSparkContext()
            train_rdd, types, shapes = _create_rdd_x_y(x, y,
                                                       self.model._feed_input_names,
                                                       self.model._feed_output_names,
                                                       sc)

            val_rdd = None
            if validation_data is not None:
                val_rdd, _, _ = _create_rdd_x_y(validation_data[0], validation_data[1],
                                                self.model._feed_input_names,
                                                self.model._feed_output_names,
                                                sc)
            names = self.model._feed_input_names + self.model._feed_output_names
            dataset = TFDataset.from_rdd(train_rdd,
                                         names=names,
                                         shapes=shapes,
                                         types=types,
                                         batch_size=batch_size if batch_size is not None else 32,
                                         val_rdd=val_rdd)
            self._fit_distributed(dataset, validation_split, epochs, **kwargs)

        else:
            self.model.fit(x=x,
                           y=y,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           **kwargs
                           )

    def _fit_distributed(self, dataset, validation_split, epochs, **kwargs):
        if not self.tf_optimizer:
            self.tf_optimizer = TFOptimizer.from_keras(self.model, dataset,
                                                       val_spilt=validation_split, **kwargs)
        else:
            self.tf_optimizer.refresh_weights()

        end_epoch = self.tf_optimizer_done_epochs + epochs
        self.tf_optimizer.optimize(MaxEpoch(end_epoch))
        self.tf_optimizer_done_epochs = end_epoch

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_per_thread=None,
                 distributed=False
                 ):
        if isinstance(x, TFDataset):
            if not x.has_batch:
                raise ValueError("The batch_per_thread of TFDataset must be " +
                                 "specified when used in KerasModel evaluate.")
            x = _standarize_feature_label_dataset(x, self.model)
            # todo check arguments
            return self._evaluate_distributed(x)
        else:
            if distributed:
                sc = getOrCreateSparkContext()
                rdd, types, shapes = _create_rdd_x_y(x, y,
                                                     self.model._feed_input_names,
                                                     self.model._feed_output_names,
                                                     sc)
                names = self.model._feed_input_names + self.model._feed_output_names
                dataset = TFDataset.from_rdd(rdd,
                                             names=names,
                                             types=types,
                                             shapes=shapes,
                                             batch_per_thread=-1 if batch_per_thread is None
                                             else batch_per_thread)
                return self._evaluate_distributed(dataset)
            else:
                return self.model.evaluate(x=x,
                                           y=y,
                                           batch_size=batch_per_thread)

    def _evaluate_distributed(self, dataset):
        predictor = TFPredictor(K.get_session(), self.metrics_tensors,
                                self.model.inputs + self.model.targets, dataset)
        result = predictor.predict()

        def elem_sum(arr1, arr2):
            result = []
            for i in range(len(arr1)):
                result.append(arr1[i] + arr2[i])
            return result

        metrics_sum = result.map(lambda x: x + [np.array(1.0)]).reduce(lambda a, b: elem_sum(a, b))
        length = len(metrics_sum) - 1
        for i in range(length):
            metrics_sum[i] /= metrics_sum[length]
        return metrics_sum[:length]

    def predict(self,
                x,
                batch_per_thread=None,
                distributed=False):

        if isinstance(x, TFDataset):
            # todo check arguments
            if not x.has_batch:
                raise ValueError("The batch_per_thread of TFDataset" +
                                 " must be specified when used in KerasModel predict.")
            x = _standarize_feature_dataset(x, self.model)
            return self._predict_distributed(x)
        else:
            if distributed:
                sc = getOrCreateSparkContext()
                rdd, types, shapes = _create_rdd_x(x, self.model._feed_input_names, sc)

                dataset = TFDataset.from_rdd(rdd,
                                             names=self.model._feed_input_names,
                                             types=types,
                                             shapes=shapes,
                                             batch_per_thread=-1 if batch_per_thread is None
                                             else batch_per_thread)
                results = self._predict_distributed(dataset).collect()
                output_num = len(self.model.outputs)
                if output_num == 1:
                    return np.stack(results)
                else:
                    predictions = []
                    for i in range(0, output_num):
                        predictions.append(np.stack([res[i] for res in results]))
                    return predictions
            else:
                return self.model.predict(x=x,
                                          batch_size=batch_per_thread)

    def _predict_distributed(self, x):
        predictor = TFPredictor.from_keras(self.model, x)
        return predictor.predict()

    def train_on_batch(self,
                       x,
                       y=None,
                       sample_weight=None,
                       class_weight=None,
                       reset_metrics=True):
        return self.model.train_on_batch(x=x,
                                         y=y,
                                         sample_weight=sample_weight,
                                         class_weight=class_weight,
                                         reset_metrics=reset_metrics)

    def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
        return self.model.test_on_batch(x=x,
                                        y=y,
                                        sample_weight=sample_weight,
                                        reset_metrics=reset_metrics)

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)


def _standarize_feature_label_dataset(dataset, model):
    input_names = model.input_names
    output_names = model.output_names

    def _process_labels(ys):
        if isinstance(ys, dict):
            return {k: np.expand_dims(y, axis=1) if y.ndim == 0 else y for k, y in ys.items()}
        elif isinstance(ys, list):
            return [np.expand_dims(y, axis=1) if y.ndim == 0 else y for y in ys]
        else:
            return np.expand_dims(ys, axis=1) if ys.ndim == 0 else ys

    def _training_reorder(x, input_names, output_names):
        assert isinstance(x, tuple)

        return _reorder(x[0], input_names) + _reorder(x[1], output_names)

    def _reorder(x, names):
        if isinstance(x, dict):
            return [x[name] for name in names]
        elif isinstance(x, list):
            return x
        else:
            return [x]

    rdd = dataset.rdd.map(lambda x: (x[0], _process_labels(x[1])))\
        .map(lambda sample: _training_reorder(sample, input_names, output_names))
    if dataset.val_rdd is not None:
        val_rdd = dataset.val_rdd.map(lambda x: (x[0], _process_labels(x[1])))\
            .map(lambda sample: _training_reorder(sample, input_names, output_names))
    else:
        val_rdd = None
    tensor_structure = _training_reorder(dataset.tensor_structure, input_names, output_names)
    new_dataset = TFDataset(rdd, tensor_structure, dataset.batch_size,
                            -1, dataset.hard_code_batch_size, val_rdd)
    new_dataset.batch_per_thread = dataset.batch_per_thread
    return new_dataset


def _standarize_feature_dataset(dataset, model):
    input_names = model.input_names

    def _reorder(x, names):
        if isinstance(x, dict):
            return [x[name] for name in names]
        elif isinstance(x, list):
            return x
        else:
            return [x]

    rdd = dataset.rdd.map(lambda sample: _reorder(sample, input_names))
    feature_schema = _reorder(dataset.tensor_structure[0], input_names)

    dataset = TFDataset(rdd, feature_schema, dataset.batch_size,
                        -1, dataset.hard_code_batch_size)
    return dataset


def _create_rdd_x_y(x, y, input_names, output_names, sc):
    x = training_utils.standardize_input_data(x, input_names,
                                              check_batch_axis=False,
                                              exception_prefix='input')
    y = training_utils.standardize_input_data(y, output_names,
                                              shapes=None, check_batch_axis=False,
                                              exception_prefix='target')

    num_samples = x[0].shape[0]
    num_inputs = len(x)
    num_targets = len(y)

    input_data = []
    for i in range(num_samples):
        inputs = []
        for j in range(num_inputs):
            inputs.append(x[j][i])

        targets = []
        for j in range(num_targets):
            if y[j][i].ndim == 0:
                targets.append(np.expand_dims(y[j][i], axis=1))
            else:
                targets.append(y[j][i])

        input_data.append(inputs + targets)

    types = [x.dtype for x in input_data[0]]
    shapes = [x.shape for x in input_data[0]]

    rdd = sc.parallelize(input_data)
    return rdd, types, shapes


def _create_rdd_x(x, input_names, sc):
    x = training_utils.standardize_input_data(x, input_names,
                                              check_batch_axis=False,
                                              exception_prefix='input')

    num_samples = x[0].shape[0]
    num_inputs = len(x)

    input_data = []
    for i in range(num_samples):
        sample = []
        for j in range(num_inputs):
            sample.append(x[j][i])

        input_data.append(sample)

    types = [x.dtype for x in input_data[0]]
    shapes = [x.shape for x in input_data[0]]

    rdd = sc.parallelize(input_data)
    return rdd, types, shapes

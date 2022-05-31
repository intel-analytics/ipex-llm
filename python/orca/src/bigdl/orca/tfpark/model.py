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

import numpy as np
from bigdl.dllib.optim.optimizer import MaxEpoch

from bigdl.orca.tfpark.utils import evaluate_string_metrics
from bigdl.dllib.utils.file_utils import enable_multi_fs_save, enable_multi_fs_load
from bigdl.dllib.nncontext import getOrCreateSparkContext
from bigdl.orca.tfpark.tf_dataset import TFNdarrayDataset, TFDataset, \
    _standarize_feature_label_dataset, check_data_compatible

from bigdl.orca.tfpark.tf_optimizer import TFOptimizer
from bigdl.orca.tfpark.tf_predictor import TFPredictor
from bigdl.dllib.utils.log4Error import *


class KerasModel(object):

    def __init__(self, model, model_dir=None, optimizer=None):
        """
        :param model: a compiled keras model
        """
        self.model = model
        self.model_dir = model_dir
        self.optimizer = optimizer
        import tensorflow as tf
        self.real_batch_size = tf.shape(self.model.inputs[0])[0]
        self.metric_tensors = {}

    def add_metric(self, tensor, name):
        self.metric_tensors[name] = tensor

    @property
    def metrics_names(self):
        return self.model.metrics_names

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    @enable_multi_fs_save
    def save_weights(self, filepath, overwrite=True, save_format=None):
        self.model.save_weights(filepath, overwrite=overwrite, save_format=save_format)

    @enable_multi_fs_load
    def load_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name=by_name)

    @enable_multi_fs_save
    def save_model(self, path, overwrite=True):
        """
        Save the model to a single HDF5 file.

        :param path: String. The path to save the model.
        :param overwrite: Boolean. Whether to silently overwrite any existing file at the target
                location
        """

        self.model.save(path, overwrite=overwrite)

    @staticmethod
    def load_model(path):
        """
        Load an existing keras model (with weights) from HDF5 file.

        :param path: String. The path to the pre-defined model.
        :return: KerasModel.
        """
        from tensorflow.python.keras import models
        keras_model = models.load_model(path)
        return KerasModel(keras_model)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            validation_data=None,
            distributed=False,
            **kwargs
            ):
        """
        Train the model for a fixed num of epochs

        Arguments:
        :param x: Input data. It could be:
            - a TFDataset object
            - A Numpy array (or array-like), or a list of arrays
               (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
        :param y: Target data. Like the input data `x`,
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a TFDataset, `y` should
          not be specified (since targets will be obtained from `x`).
        :param batch_size: Integer or `None`.
            Number of samples per gradient update.
            If `x` is a TFDataset, you do not need to specify batch_size.
        :param epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
        :param validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` could be:
              - tuple `(x_val, y_val)` of Numpy arrays or tensors
        :param distributed: Boolean. Whether to do prediction in distributed mode or local mode.
                     Default is True. In local mode, x must be a Numpy array.
        """
        if isinstance(x, TFDataset):
            # todo check arguments
            invalidInputError(validation_data is None,
                              "validation_data must be None when"
                              " using TFDataset as input, please use set"
                              " the validation data in TFDataset")
            if not x.has_batch:
                invalidInputError(False,
                                  "The batch_size of TFDataset must be " +
                                  "specified when used in KerasModel fit.")
            self._fit_distributed(x, epochs, **kwargs)

        elif distributed:
            dataset = TFDataset.from_ndarrays((x, y), val_tensors=validation_data,
                                              batch_size=batch_size)
            self._fit_distributed(dataset, epochs, **kwargs)

        else:
            self.model.fit(x=x,
                           y=y,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=validation_data,
                           **kwargs
                           )

    def _fit_distributed(self, dataset, epochs, **kwargs):
        self.tf_optimizer = TFOptimizer.from_keras(self.model, dataset,
                                                   model_dir=self.model_dir,
                                                   metrics=self.metric_tensors,
                                                   optimizer=self.optimizer,
                                                   **kwargs)

        self.tf_optimizer.optimize(MaxEpoch(epochs))

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_per_thread=None,
                 distributed=False
                 ):
        """
        Evaluate a model on a given dataset

        :param x: Input data. It could be:
            - a TFDataset object
            - A Numpy array (or array-like), or a list of arrays
               (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
        :param y: Target data. Like the input data `x`,
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a TFDataset, `y` should
          not be specified (since targets will be obtained from `x`).
        :param batch_per_thread:
          The default value is 1.
          When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
          When distributed is False the total batch size is batch_per_thread * numOfCores.
        :param distributed: Boolean. Whether to do prediction in distributed mode or local mode.
                     Default is True. In local mode, x must be a Numpy array.
        """
        if isinstance(x, TFDataset):
            if not x.has_batch:
                invalidInputError(False,
                                  "The batch_per_thread of TFDataset must be " +
                                  "specified when used in KerasModel evaluate.")
            if isinstance(x, TFNdarrayDataset):
                x = _standarize_feature_label_dataset(x, self.model)
            # todo check arguments
            check_data_compatible(x, self.model, mode="evaluate")

            return self._evaluate_distributed(x)
        else:
            if distributed:
                dataset = TFDataset.from_ndarrays((x, y),
                                                  batch_per_thread=-1 if batch_per_thread is None
                                                  else batch_per_thread
                                                  )
                dataset = _standarize_feature_label_dataset(dataset, self.model)
                return self._evaluate_distributed(dataset)
            else:
                results = self.model.evaluate(x=x,
                                              y=y,
                                              batch_size=batch_per_thread)
                results = dict(zip(self.metrics_names, results))
                return results

    def _evaluate_distributed(self, dataset):

        import tensorflow.keras.backend as K

        if hasattr(self.model, "targets"):
            model_targets = self.model.targets
        else:
            model_targets = self.model._targets

        return evaluate_string_metrics(sess=K.get_session(),
                                       string_metrics=self.metrics_names,
                                       dataset=dataset,
                                       inputs=self.model.inputs + model_targets,
                                       targets=model_targets,
                                       outputs=self.model.outputs,
                                       loss=self.model.total_loss)

    def predict(self,
                x,
                batch_per_thread=None,
                distributed=False):

        """
        Use a model to do prediction.

        :param x: Input data. It could be:
            - a TFDataset object
            - A Numpy array (or array-like), or a list of arrays
               (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
        :param batch_per_thread:
          The default value is 1.
          When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
          When distributed is False the total batch size is batch_per_thread * numOfCores.
        :param distributed: Boolean. Whether to do prediction in distributed mode or local mode.
                     Default is True. In local mode, x must be a Numpy array.
        """

        if isinstance(x, TFDataset):
            # todo check arguments
            if not x.has_batch:
                invalidInputError(False,
                                  "The batch_per_thread of TFDataset" +
                                  " must be specified when used in KerasModel predict.")
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


def _create_rdd_x_y(x, y, input_names, output_names, sc):
    from tensorflow.python.keras.engine import training_utils
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

        input_data.append((inputs, targets))

    x_meta = dict([(input_names[i],
                    (input_data[0][0][i].dtype, input_data[0][0][i].shape))
                   for i in range(len(input_names))])

    y_meta = dict([(output_names[i],
                    (input_data[0][1][i].dtype, input_data[0][1][i].shape))
                   for i in range(len(input_names))])

    rdd = sc.parallelize(input_data)
    return rdd, x_meta, y_meta


def _create_rdd_x(x, input_names, sc):
    from tensorflow.python.keras.engine import training_utils
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

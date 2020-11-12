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

import numpy as np
from bigdl.optim.optimizer import MaxEpoch

from zoo.tfpark.utils import evaluate_string_metrics
from zoo.common import load_from_file
from zoo.common import save_file
from zoo.common.nncontext import getOrCreateSparkContext
from zoo.tfpark.tf_dataset import TFNdarrayDataset, TFDataset

from zoo.tfpark.tf_optimizer import TFOptimizer
from zoo.tfpark.tf_predictor import TFPredictor


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

    def save_weights(self, filepath, overwrite=True, save_format=None):

        def save_func(file_path):
            self.model.save_weights(file_path, overwrite, save_format)
        save_file(save_func, filepath)

    def load_weights(self, filepath, by_name=False):

        def load_func(file_path):
            self.model.load_weights(file_path, by_name)
        load_from_file(load_func, filepath)

    def save_model(self, path, overwrite=True):
        """
        Save the model to a single HDF5 file.

        :param path: String. The path to save the model.
        :param overwrite: Boolean. Whether to silently overwrite any existing file at the target
                location
        """
        def save_func(file_path, over_write=True):
            self.model.save(file_path, overwrite=over_write)
        save_file(save_func, path, over_write=overwrite)

    @staticmethod
    def load_model(path):
        """
        Load an existing keras model (with weights) from HDF5 file.

        :param path: String. The path to the pre-defined model.
        :return: KerasModel.
        """
        from tensorflow.python.keras import models

        def load_func(file_path):
            return models.load_model(file_path)

        keras_model = load_from_file(load_func, path)
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
            assert validation_data is None, "validation_data must be None when " \
                                            "using TFDataset as input, please " \
                                            "use set the validation data in TFDataset"
            if not x.has_batch:
                raise ValueError("The batch_size of TFDataset must be " +
                                 "specified when used in KerasModel fit.")
            if isinstance(x, TFNdarrayDataset):
                x = _standarize_feature_label_dataset(x, self.model)
            self._fit_distributed(x, epochs, **kwargs)

        elif distributed:
            dataset = TFDataset.from_ndarrays((x, y), val_tensors=validation_data,
                                              batch_size=batch_size)
            dataset = _standarize_feature_label_dataset(dataset, self.model)
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
                raise ValueError("The batch_per_thread of TFDataset must be " +
                                 "specified when used in KerasModel evaluate.")
            if isinstance(x, TFNdarrayDataset):
                x = _standarize_feature_label_dataset(x, self.model)
            # todo check arguments
            return self._evaluate_distributed(x)
        else:
            if distributed:
                dataset = TFDataset.from_ndarrays((x, y),
                                                  batch_per_thread=-1 if batch_per_thread is None
                                                  else batch_per_thread
                                                  )
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
                raise ValueError("The batch_per_thread of TFDataset" +
                                 " must be specified when used in KerasModel predict.")
            if isinstance(x, TFNdarrayDataset):
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
            return {k: np.expand_dims(y, axis=-1) if y.ndim == 0 else y for k, y in ys.items()}
        elif isinstance(ys, list):
            return [np.expand_dims(y, axis=-1) if y.ndim == 0 else y for y in ys]
        elif isinstance(ys, tuple):
            return tuple([np.expand_dims(y, axis=-1) if y.ndim == 0 else y for y in ys])
        else:
            return np.expand_dims(ys, axis=-1) if ys.ndim == 0 else ys

    def _training_reorder(x, input_names, output_names):
        assert isinstance(x, tuple)

        return (_reorder(x[0], input_names), _reorder(x[1], output_names))

    def _reorder(x, names):
        if isinstance(x, dict):
            return [x[name] for name in names]
        elif isinstance(x, list) or isinstance(x, tuple):
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
    new_dataset = TFNdarrayDataset(rdd, tensor_structure, dataset.batch_size,
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
        elif isinstance(x, tuple):
            return list(x)
        return [x]

    rdd = dataset.rdd.map(lambda sample: _reorder(sample, input_names))
    feature_schema = _reorder(dataset.tensor_structure[0], input_names)

    dataset = TFNdarrayDataset(rdd, feature_schema, dataset.batch_size,
                               -1, dataset.hard_code_batch_size)
    return dataset


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

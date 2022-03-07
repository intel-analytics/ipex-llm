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

from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.metric.forecast_metrics import Evaluator
import keras


class BaseTF2Forecaster(Forecaster):
    def __init__(self, **kwargs):
        self.fitted = False
        self.internal = self.model_creator({**self.model_config})

    def fit(self, data, validation_data=None, epochs=1,
            batch_size=32, shuffle=True, verbose='auto'):
        """
        Fit(Train) the forecaster.

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.

        :params validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
        :params epochs: Number of epochs you want to train. The value defaults to 1.
        :params batch_size: Number of batch size you want to train. The value defaults to 32.
        :params shuffle: whether to shuffle the training data before each epoch.
                This argument is ignored when x is a generator or an object of tf.data.Dataset.
                The value defaults to True.
        :params verbose: 'auto', 0, 1, or 2. Verbosity mode.  0 = silent, 1 = progress bar,
                2 = one line per epoch. 'auto' defaults to 1 for most cases,
                The value defaults to 'auto'.
        """

        self.internal.fit(x=data[0], y=data[1],
                          validation_data=validation_data,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          verbose=verbose)
        self.fitted = True

    def predict(self, data, batch_size=32, quantize=False):
        """
        :params data: The data support following formats:

                | 1. a numpy ndarray x:
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.

        :params batch_size: predict batch size. The value will not affect evaluate
                result but will affect resources cost(e.g. memory and time).
        :params quantize: # TODO will support.
        """
        if not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        yhat = self.internal.predict(data, batch_size=batch_size)
        return yhat

    def evaluate(self, data, batch_size=32, multioutput="raw_values", quantize=False):
        """
        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        :params data: The data support following formats:

                | 1. a numpy ndarray tuple (x, y):
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.
                | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
                | should be the same as future_seq_len and output_feature_num.

        :params batch_size: evaluate batch size. The value will not affect evaluate
                result but will affect resources cost(e.g. memory and time).
        :params multioutput_value: Defines aggregating of multiple output values.
                String in ['raw_values', 'uniform_average']. The value defaults to
                'raw_values'.The param is only effective when the forecaster is a
                non-distribtued version.
        :params quantize: # TODO will support.
        """
        if not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        yhat = self.internal.predict(data[0], batch_size=batch_size)

        aggregate = 'mean' if multioutput == 'uniform_average' else None
        return Evaluator.evaluate(self.metrics, data[1], yhat, aggregate=aggregate)

    def save(self, checkpoint_file, quantize_checkpoint_file=None):
        """
        Save the forecaster.

        :params checkpoint_file: The location you want to save the forecaster.
        :params quantize_checkpoint_file: # TODO will support.
        """
        if not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        self.internal.save(checkpoint_file)

    def load(self, checkpoint_file, quantize_checkpoint_file=None):
        """
        Load the forecaster.

        :params checkpoint_file: The checkpoint file location you want to load the forecaster.
        :params quantize_checkpoint_file: # TODO will support.
        """
        self.internal = keras.models.load_model(checkpoint_file)
        self.fitted = True

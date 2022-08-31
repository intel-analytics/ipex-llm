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
from bigdl.chronos.data import TSDataset
from bigdl.chronos.metric.forecast_metrics import Evaluator
import keras
import tensorflow as tf
import numpy as np


class BaseTF2Forecaster(Forecaster):
    def __init__(self, **kwargs):
        self.fitted = False
        self.internal = self.model_creator({**self.model_config})
        # TF seed can't be set to None, just to be consistent with torch.
        if self.seed:
            from tensorflow.keras.utils import set_random_seed
            set_random_seed(seed=self.seed)

    def fit(self, data, epochs=1, batch_size=32):
        """
        Fit(Train) the forecaster.

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. a tf.data.Dataset:
               | A TFDataset instance which contains x and y with same shape as the tuple.
               | x's shape is (num_samples, lookback, feature_dim),
               | y's shape is (num_samples, horizon, target_dim).
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance.
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a tfdataset,
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :params epochs: Number of epochs you want to train. The value defaults to 1.
        :params batch_size: Number of batch size you want to train. The value defaults to 32.
                Do not specify the batch_size, if your data in the form of tf.data datasets.
        """
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'],
                          horizon=self.model_config['future_seq_len'])
            data = data.to_tf_dataset(shuffle=True, batch_size=batch_size)
        if isinstance(data, tuple):
            self.internal.fit(x=data[0], y=data[1], epochs=epochs, batch_size=batch_size)
        else:
            self.internal.fit(x=data, epochs=epochs)
        self.fitted = True

    def predict(self, data, batch_size=32):
        """
        :params data: The data support following formats:

                | 1. a numpy ndarray x:
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.
                | 2. a tfdataset
                | A TFDataset instance which contains x and y with same shape as the tuple.
                | the tfdataset needs to return at least x in each iteration
                | with the shape as following:
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.
                | If returns x and y only get x.
                | 3. A bigdl.chronos.data.tsdataset.TSDataset instance
                | Forecaster will automatically process the TSDataset.
                | By default, TSDataset will be transformed to a tfdataset,
                | Users may call `roll` on the TSDataset before calling `fit`
                | Then the training speed will be faster but will consume more memory.

        :params batch_size: predict batch size. The value will not affect evaluate
                result but will affect resources cost(e.g. memory and time).
                The value default to 32. If set to None,
                the model will be used directly for inference.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling predict!")
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'],
                          horizon=self.model_config['future_seq_len'])
            data = data.to_tf_dataset(shuffle=False, batch_size=batch_size)

        if batch_size or isinstance(data, tf.data.Dataset):
            yhat = self.internal.predict(data, batch_size=batch_size)
        else:
            yhat = self.internal(data, training=False).numpy()
        return yhat

    def evaluate(self, data, batch_size=32, multioutput="raw_values"):
        """
        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from bigdl.chronos.metric.forecaster_metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :params data: The data support following formats:

                | 1. a numpy ndarray tuple (x, y):
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.
                | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
                | should be the same as future_seq_len and output_feature_num.
                | 2. a tfdataset
                | A TFDataset instance which contains x and y with same shape as the tuple.
                | x's shape is (num_samples, lookback, feature_dim),
                | y's shape is (num_samples, horizon, target_dim).
                | 3. A bigdl.chronos.data.tsdataset.TSDataset instance
                | Forecaster will automatically process the TSDataset.
                | By default, TSDataset will be transformed to a tfdataset,
                | Users may call `roll` on the TSDataset before calling `fit`
                | Then the training speed will be faster but will consume more memory.

        :params batch_size: evaluate batch size. The value will not affect evaluate
                result but will affect resources cost(e.g. memory and time).
        :params multioutput_value: Defines aggregating of multiple output values.
                String in ['raw_values', 'uniform_average']. The value defaults to
                'raw_values'.The param is only effective when the forecaster is a
                non-distribtued version.

        :return: A list of evaluation results. Each item represents a metric.
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling evaluate!")
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'],
                          horizon=self.model_config['future_seq_len'])
            data = data.to_tf_dataset(shuffle=False, batch_size=batch_size)

        if isinstance(data, tuple):
            input_data, target = data
        else:
            input_data = data
            target = np.asarray(tuple(map(lambda x: x[1], data.as_numpy_iterator())))
        yhat = self.internal.predict(input_data, batch_size=batch_size)

        aggregate = 'mean' if multioutput == 'uniform_average' else None
        return Evaluator.evaluate(self.metrics, y_true=target, y_pred=yhat, aggregate=aggregate)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :params checkpoint_file: The location you want to save the forecaster.
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling save!")
        self.internal.save(checkpoint_file)

    def load(self, checkpoint_file):
        """
        Load the forecaster.

        :params checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.internal = keras.models.load_model(checkpoint_file,
                                                custom_objects=self.custom_objects_config)
        self.fitted = True

    @classmethod
    def from_tsdataset(cls, tsdataset, past_seq_len=None, future_seq_len=None, **kwargs):
        """
        Build a Forecaster Model

        :param tsdataset: A bigdl.chronos.data.tsdataset.TSDataset instance.
        :param past_seq_len:  Specify history time step (i.e. lookback)
               Do not specify the 'past_seq_len' if your tsdataset has called
               the 'TSDataset.roll' method or 'TSDataset.to_tf_dataset'.
        :param future_seq_len: Specify output time step (i.e. horizon)
               Do not specify the 'future_seq_len' if your tsdataset has called
               the 'TSDataset.roll' method or 'TSDataset.to_tf_dataset'.
        :param kwargs: Specify parameters of Forecaster,
               e.g. loss and optimizer, etc.
               More info, please refer to Forecaster.__init__ methods.

        :return: A Forecaster Model
        """
        from bigdl.nano.utils.log4Error import invalidInputError

        def check_time_steps(tsdataset, past_seq_len, future_seq_len):
            if tsdataset.lookback and past_seq_len:
                future_seq_len = future_seq_len if isinstance(future_seq_len, int)\
                    else max(future_seq_len)
                return tsdataset.lookback == past_seq_len and tsdataset.horizon == future_seq_len
            return True

        invalidInputError(not tsdataset._has_generate_agg_feature,
                          "We will add support for 'gen_rolling_feature' method later.")

        if tsdataset.lookback:
            past_seq_len = tsdataset.lookback
            future_seq_len = tsdataset.horizon if isinstance(tsdataset.horizon, int) \
                else max(tsdataset.horizon)
            output_feature_num = len(tsdataset.roll_target)
            input_feature_num = len(tsdataset.roll_feature) + output_feature_num
        elif past_seq_len and future_seq_len:
            past_seq_len = past_seq_len if isinstance(past_seq_len, int)\
                else tsdataset.get_cycle_length()
            future_seq_len = future_seq_len if isinstance(future_seq_len, int) \
                else max(future_seq_len)
            output_feature_num = len(tsdataset.target_col)
            input_feature_num = len(tsdataset.feature_col) + output_feature_num
        else:
            invalidInputError(False,
                              "Forecaster needs 'past_seq_len' and 'future_seq_len' "
                              "to specify the history time step of training.")

        invalidInputError(check_time_steps(tsdataset, past_seq_len, future_seq_len),
                          "tsdataset already has history time steps and "
                          "differs from the given past_seq_len and future_seq_len "
                          "Expected past_seq_len and future_seq_len to be "
                          f"{tsdataset.lookback, tsdataset.horizon}, "
                          f"but found {past_seq_len, future_seq_len}.",
                          fixMsg="Do not specify past_seq_len and future seq_len "
                          "or call tsdataset.roll method again and specify time step")

        return cls(past_seq_len=past_seq_len,
                   future_seq_len=future_seq_len,
                   input_feature_num=input_feature_num,
                   output_feature_num=output_feature_num,
                   **kwargs)

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

import warnings
import keras
import tensorflow as tf
import numpy as np
from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.data import TSDataset
from bigdl.chronos.metric.forecast_metrics import Evaluator
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.chronos.forecaster.tf.utils import np_to_data_creator, tsdata_to_data_creator,\
    np_to_xshards, xshard_to_np, np_to_tfdataset


class BaseTF2Forecaster(Forecaster):
    def __init__(self, **kwargs):
        if self.seed:
            # TF seed can't be set to None, just to be consistent with torch.
            from tensorflow.keras.utils import set_random_seed
            set_random_seed(seed=self.seed)
        if self.distributed:
            from bigdl.orca.learn.tf2.estimator import Estimator
            self.internal = Estimator.from_keras(model_creator=self.model_creator,
                                                 config=self.model_config,
                                                 workers_per_node=self.workers_per_node,
                                                 backend=self.remote_distributed_backend)
        else:
            self.internal = self.model_creator({**self.model_config})

        self.fitted = False

    def fit(self, data, epochs=1, batch_size=32):
        """
        Fit(Train) the forecaster.

        :param data: The data support following formats:

               | 1. A numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. A tf.data.Dataset:
               | A TFDataset instance which contains x and y with same shape as the tuple.
               | x's shape is (num_samples, lookback, feature_dim),
               | y's shape is (num_samples, horizon, target_dim).
               | If set distributed to True, we do not recommend using tf.data.Dataset,
               | please replace with tsdataset or numpy.ndarray.
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

        if self.distributed:
            if isinstance(data, tuple):
                data = np_to_data_creator(data, shuffle=True)
            if isinstance(data, TSDataset):
                data = tsdata_to_data_creator(data, shuffle=True)
            from bigdl.nano.utils.log4Error import invalidInputError
            invalidInputError(not isinstance(data, tf.data.Dataset),
                              "tf.data.Dataset is not supported, "
                              "please replace with numpy.ndarray or TSDataset instance.")
            # for cluster mode
            from bigdl.orca.common import OrcaContext
            sc = OrcaContext.get_spark_context().getConf()
            num_nodes = 1 if sc.get('spark.master').startswith('local') \
                else int(sc.get('spark.executor.instances'))
            if batch_size % self.workers_per_node != 0:
                from bigdl.nano.utils.log4Error import invalidInputError
                invalidInputError(False,
                                  "Please make sure that batch_size can be divisible by "
                                  "the product of worker_per_node and num_nodes, "
                                  f"but 'batch_size' is {batch_size}, 'workers_per_node' "
                                  f"is {self.workers_per_node}, 'num_nodes' is {num_nodes}")
            batch_size //= (self.workers_per_node * num_nodes)
            self.internal.fit(data, epochs=epochs, batch_size=batch_size)
        else:
            if isinstance(data, tuple):
                self.internal.fit(x=data[0], y=data[1], epochs=epochs, batch_size=batch_size)
            else:
                if isinstance(data, TSDataset):
                    data = data.to_tf_dataset(shuffle=True, batch_size=batch_size)
                self.internal.fit(data, epochs=epochs)
        self.fitted = True

    def predict(self, data, batch_size=32):
        """
        :params data: The data support following formats:

                | 1. A numpy ndarray x:
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.
                |
                | 2. A tfdataset
                | A TFDataset instance which contains x and y with same shape as the tuple.
                | the tfdataset needs to return at least x in each iteration
                | with the shape as following:
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.
                | If returns x and y only get x.
                |
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
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'],
                          horizon=self.model_config['future_seq_len'])

        invalidInputError(self.fitted,
                          "You must call fit or restore first before calling predict!")

        if self.distributed:
            if isinstance(data, np.ndarray):
                data = np_to_xshards(data, self.workers_per_node)
            if isinstance(data, TSDataset):
                input_data, _ = data.to_numpy()
                data = np_to_xshards(input_data, self.workers_per_node)
            invalidInputError(not isinstance(data, tf.data.Dataset),
                              "prediction on tf.data.Dataset will be supported in future.")
            yhat = self.internal.predict(data, batch_size=batch_size)

            # pytorch and tf2 have different behavior, when `future_seq_len` and
            # `output_feature_num` are equal to 1, they will not squeeze data.
            yhat = xshard_to_np(yhat, mode="yhat")
        else:
            if isinstance(data, TSDataset):
                data = data.to_tf_dataset(batch_size, batch_size)
            if batch_size or isinstance(data, tf.data.Dataset):
                yhat = self.internal.predict(data)
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

                | 1. A numpy ndarray tuple (x, y):
                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
                | should be the same as past_seq_len and input_feature_num.
                | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
                | should be the same as future_seq_len and output_feature_num.
                |
                | 2. A tf.data.Dataset:
                | A TFDataset instance which contains x and y with same shape as the tuple.
                | x's shape is (num_samples, lookback, feature_dim),
                | y's shape is (num_samples, horizon, target_dim).
                | If set distributed to True, we do not recommend using tf.data.Dataset,
                | please replace with tsdataset or numpy.ndarray.
                |
                | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
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
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'],
                          horizon=self.model_config['future_seq_len'])

        invalidInputError(self.fitted,
                          "You must call fit or restore first before calling evaluate!")

        if self.distributed:
            if isinstance(data, tuple):
                data = np_to_data_creator(data, shuffle=False)
            if isinstance(data, TSDataset):
                data = tsdata_to_data_creator(data, shuffle=False)
            invalidInputError(not isinstance(data, tf.data.Dataset),
                              "tf.data.Dataset is not supported, "
                              "please replace with numpy.ndarray or TSDataset instance.")
            return self.internal.evaluate(data, batch_size=batch_size)
        else:
            if isinstance(data, tuple):
                input_data, target = data
            else:
                input_data = data
                target = np.asarray(tuple(map(lambda x: x[1], data.as_numpy_iterator())))
            yhat = self.internal.predict(input_data, batch_size=batch_size)

            aggregate = 'mean' if multioutput == 'uniform_average' else None
            return Evaluator.evaluate(self.metrics, y_true=target, y_pred=yhat, aggregate=aggregate)

    def to_local(self):
        """
        Transform a distributed forecaster to a local (non-distributed) one.

        you need to call .to_local() and transform the forecaster to a non-
        distributed one.

        """
        if not self.distributed:
            warnings.warn("The forecaster has become local.")

        model = self.internal.get_model()
        self.internal.shutdown()
        self.internal = model

        self.distributed = False

    def get_model(self):
        """
        Returns the learned Keras model.

        :return: a keras model instance
        """
        if self.distributed:
            return self.internal.get_model()
        else:
            return self.internal

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :params checkpoint_file: The location you want to save the forecaster.
        """
        invalidInputError(self.fitted,
                          "You must call fit or restore first before calling save!")

        self.internal.save(checkpoint_file)

    def load(self, checkpoint_file):
        """
        Load the forecaster.

        :params checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        if self.distributed:
            self.internal.load(checkpoint_file)
        else:
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

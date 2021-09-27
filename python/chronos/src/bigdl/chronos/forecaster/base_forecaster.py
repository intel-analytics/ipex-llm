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

from zoo.chronos.forecaster.abstract import Forecaster
from zoo.orca.data.shard import SparkXShards
from zoo.chronos.forecaster.utils import\
    np_to_creator, set_pytorch_seed, check_data, xshard_to_np, np_to_xshard
import numpy as np
from zoo.orca.learn.pytorch.estimator import Estimator
from zoo.orca.learn.metrics import MSE, MAE

ORCA_METRICS = {"mse": MSE, "mae": MAE}


class BasePytorchForecaster(Forecaster):
    '''
    Forecaster base model for lstm, mtnet, seq2seq and tcn forecasters.
    '''
    def __init__(self, **kwargs):
        if self.distributed:
            def model_creator_orca(config):
                set_pytorch_seed(self.seed)
                model = self.model_creator({**self.config, **self.data_config})
                model.train()
                return model
            self.internal = Estimator.from_torch(model=model_creator_orca,
                                                 optimizer=self.optimizer_creator,
                                                 loss=self.loss_creator,
                                                 metrics=[ORCA_METRICS[name]()
                                                          for name in self.metrics],
                                                 backend=self.distributed_backend,
                                                 use_tqdm=True,
                                                 config={"lr": self.lr},
                                                 workers_per_node=self.workers_per_node)
        else:
            set_pytorch_seed(self.seed)
            self.internal = self.local_model(check_optional_config=False)

    def fit(self, data, epochs=1, batch_size=32):
        # TODO: give an option to close validation during fit to save time.
        """
        Fit(Train) the forecaster.

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.

        :param epochs: Number of epochs you want to train. The value defaults to 1.
        :param batch_size: Number of batch size you want to train. The value defaults to 32.

        :return: Evaluation results on data.
        """

        # input adaption
        self.config["batch_size"] = batch_size

        # input transform
        if isinstance(data, tuple) and self.distributed:
            data = np_to_creator(data)
        if isinstance(data, SparkXShards) and not self.distributed:
            data = xshard_to_np(data)

        # fit on internal
        if self.distributed:
            return self.internal.fit(data=data,
                                     epochs=epochs,
                                     batch_size=batch_size)
        else:
            check_data(data[0], data[1], self.data_config)
            return self.internal.fit_eval(data=data,
                                          validation_data=data,
                                          epochs=epochs,
                                          metric=self.metrics[0],  # only use the first metric
                                          **self.config)

    def predict(self, data, batch_size=32):
        """
        Predict using a trained forecaster.

        if you want to predict on a single node(which is common practice), please call
        .to_local().predict(x, ...)

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x}, where x's shape
               | should follow the shape stated before.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).

        :return: A numpy array with shape (num_samples, horizon, target_dim)
                 if data is a numpy ndarray. A xshard item with format {‘prediction’: result},
                 where result is a numpy array with shape (num_samples, horizon, target_dim)
                 if data is a xshard item.
        """
        # data transform
        is_local_data = isinstance(data, np.ndarray)
        if is_local_data and self.distributed:
            data = np_to_xshard(data)
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="predict")

        if self.distributed:
            yhat = self.internal.predict(data, batch_size=batch_size)
            if is_local_data:
                expand_dim = []
                if self.data_config["future_seq_len"] == 1:
                    expand_dim.append(1)
                if self.data_config["output_feature_num"] == 1:
                    expand_dim.append(2)
                yhat = xshard_to_np(yhat, mode="yhat", expand_dim=expand_dim)
            return yhat
        else:
            if not self.internal.model_built:
                raise RuntimeError("You must call fit or restore first before calling predict!")
            yhat = self.internal.predict(data, batch_size=batch_size)
            if not is_local_data:
                yhat = np_to_xshard(yhat, prefix="prediction")
            return yhat

    def predict_with_onnx(self, data, batch_size=32, dirname=None):
        """
        Predict using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_onnx is valid and Forecaster will
        automatically build an onnxruntime session with default settings.

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        """
        if self.distributed:
            raise NotImplementedError("ONNX inference has not been supported for distributed\
                                       forecaster. You can call .to_local() to transform the\
                                       forecaster to a non-distributed version.")
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict_with_onnx(data, batch_size=batch_size, dirname=dirname)

    def evaluate(self, data, batch_size=32, multioutput="raw_values"):
        """
        Evaluate using a trained forecaster.

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        if you want to evaluate on a single node(which is common practice), please call
        .to_local().evaluate(data, ...)

        >>> from zoo.orca.automl.metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.

        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.The param is only effective when the forecaster is a
               non-distribtued version.

        :return: A list of evaluation results. Each item represents a metric.
        """
        # data transform
        is_local_data = isinstance(data, tuple)
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="fit")
        if self.distributed:
            if is_local_data:
                return self.internal.evaluate(data=np_to_creator(data),
                                              batch_size=batch_size)
            else:
                return self.internal.evaluate(data=data,
                                              batch_size=batch_size)
        else:
            if not self.internal.model_built:
                raise RuntimeError("You must call fit or restore first before calling evaluate!")
            return self.internal.evaluate(data[0], data[1], metrics=self.metrics,
                                          multioutput=multioutput, batch_size=batch_size)

    def evaluate_with_onnx(self, data,
                           batch_size=32,
                           dirname=None,
                           multioutput="raw_values"):
        """
        Evaluate using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_onnx is valid and Forecaster will
        automatically build an onnxruntime session with default settings.

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from zoo.orca.automl.metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.

        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.

        :return: A list of evaluation results. Each item represents a metric.
        """
        if self.distributed:
            raise NotImplementedError("ONNX inference has not been supported for distributed\
                                       forecaster. You can call .to_local() to transform the\
                                       forecaster to a non-distributed version.")
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate_with_onnx(data[0], data[1],
                                                metrics=self.metrics,
                                                dirname=dirname,
                                                multioutput=multioutput,
                                                batch_size=batch_size)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        Please note that if you only want the pytorch model or onnx model
        file, you can call .get_model() or .export_onnx_file(). The checkpoint
        file generated by .save() method can only be used by .load().

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if self.distributed:
            self.internal.save(checkpoint_file)
        else:
            if not self.internal.model_built:
                raise RuntimeError("You must call fit or restore first before calling save!")
            self.internal.save(checkpoint_file)

    def load(self, checkpoint_file):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        if self.distributed:
            self.internal.load(checkpoint_file)
        else:
            self.internal.restore(checkpoint_file)

    def to_local(self):
        """
        Transform a distributed forecaster to a local (non-distributed) one.

        Common practice is to use distributed training (fit) and predict/
        evaluate with onnx or other frameworks on a single node. To do so,
        you need to call .to_local() and transform the forecaster to a non-
        distributed one.

        The optimizer is refreshed, incremental training after to_local
        might have some problem.

        :return: a forecaster instance.
        """
        # TODO: optimizer is refreshed, which is not reasonable
        if not self.distributed:
            raise RuntimeError("The forecaster has become local.")
        model = self.internal.get_model()
        state = {
            "config": {**self.data_config, **self.config},
            "model": model.state_dict(),
            "optimizer": self.optimizer_creator(model, {"lr": self.config["lr"]}).state_dict(),
        }
        self.internal.shutdown()
        self.internal = self.local_model(check_optional_config=False)
        self.internal.load_state_dict(state)
        self.distributed = False
        return self

    def get_model(self):
        """
        Returns the learned PyTorch model.

        :return: a pytorch model instance
        """
        if self.distributed:
            return self.internal.get_model()
        else:
            return self.internal.model

    def build_onnx(self, thread_num=None, sess_options=None):
        '''
        Build onnx model to speed up inference and reduce latency.
        The method is Not required to call before predict_with_onnx,
        evaluate_with_onnx or export_onnx_file.
        It is recommended to use when you want to:

        | 1. Strictly control the thread to be used during inferencing.
        | 2. Alleviate the cold start problem when you call predict_with_onnx
             for the first time.

        :param thread_num: int, the num of thread limit. The value is set to None by
               default where no limit is set.
        :param sess_options: an onnxruntime.SessionOptions instance, if you set this
               other than None, a new onnxruntime session will be built on this setting
               and ignore other settings you assigned(e.g. thread_num...).

        Example:
            >>> # to pre build onnx sess
            >>> forecaster.build_onnx(thread_num=1)  # build onnx runtime sess for single thread
            >>> pred = forecaster.predict_with_onnx(data)
            >>> # ------------------------------------------------------
            >>> # directly call onnx related method is also supported
            >>> pred = forecaster.predict_with_onnx(data)
        '''
        import onnxruntime
        if sess_options is not None and not isinstance(sess_options, onnxruntime.SessionOptions):
            raise RuntimeError("sess_options should be an onnxruntime.SessionOptions instance"
                               f", but f{type(sess_options)}")
        if self.distributed:
            raise NotImplementedError("build_onnx has not been supported for distributed\
                                       forecaster. You can call .to_local() to transform the\
                                       forecaster to a non-distributed version.")
        import torch
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.internal._build_onnx(dummy_input,
                                  dirname=None,
                                  thread_num=thread_num,
                                  sess_options=None)

    def export_onnx_file(self, dirname):
        """
        Save the onnx model file to the disk.

        :param dirname: The dir location you want to save the onnx file.
        """
        if self.distributed:
            raise NotImplementedError("export_onnx_file has not been supported for distributed\
                                       forecaster. You can call .to_local() to transform the\
                                       forecaster to a non-distributed version.")
        import torch
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.internal._build_onnx(dummy_input, dirname)

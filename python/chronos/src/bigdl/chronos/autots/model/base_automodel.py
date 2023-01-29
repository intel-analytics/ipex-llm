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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import json
from bigdl.chronos.autots.utils import recalculate_n_sampling
import warnings


class BaseAutomodel:

    def __init__(self, **kwargs):
        if self.backend.startswith("torch"):
            from bigdl.orca.automl.model.base_pytorch_model import PytorchModelBuilder
            self._DEFAULT_BEST_MODEL_DIR = "best_model.ckpt"
            self._DEFAULT_BEST_CONFIG_DIR = "best_config.json"
            model_builder = PytorchModelBuilder(model_creator=self._model_creator,
                                                optimizer_creator=self.optimizer,
                                                loss_creator=self.loss)
        elif self.backend.startswith("keras"):
            from bigdl.orca.automl.model.base_keras_model import KerasModelBuilder
            self.search_space.update({"optimizer": self.optimizer, "loss": self.loss})
            model_builder = KerasModelBuilder(model_creator=self._model_creator)
            self._DEFAULT_BEST_MODEL_DIR = "best_keras_model.ckpt"
            self._DEFAULT_BEST_CONFIG_DIR = "best_keras_config.json"

        from bigdl.orca.automl.auto_estimator import AutoEstimator
        self.auto_est = AutoEstimator(model_builder, **self._auto_est_config)
        self.best_model = None

    def fit(self,
            data,
            epochs=1,
            batch_size=32,
            validation_data=None,
            metric_threshold=None,
            n_sampling=1,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            ):
        """
        Automatically fit the model and search for the best hyper parameters.

        :param data: train data.
               data can be a tuple of ndarrays or a PyTorch DataLoader
               or a function that takes a config dictionary as parameter and returns a
               PyTorch DataLoader.
        :param epochs: Max number of epochs to train in each trial. Defaults to 1.
               If you have also set metric_threshold, a trial will stop if either it has been
               optimized to the metric_threshold or it has been trained for {epochs} epochs.
        :param batch_size: Int or hp sampling function from an integer space. Training batch size.
               It defaults to 32.
        :param validation_data: Validation data. Validation data type should be the same as data.
        :param metric_threshold: a trial will be terminated when metric threshold is met.
        :param n_sampling: Number of trials to evaluate in total. Defaults to 1.
               If hp.grid_search is in search_space, the grid will be run n_sampling of trials
               and round up n_sampling according to hp.grid_search.
               If this is -1, (virtually) infinite samples are generated
               until a stopping condition is met.
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt").
        :param search_alg_params: extra parameters for searcher algorithm besides search_space,
               metric and searcher mode.
        :param scheduler: str, all supported scheduler provided by ray tune.
        :param scheduler_params: parameters for scheduler.
        """
        self.search_space["batch_size"] = batch_size
        n_sampling = recalculate_n_sampling(self.search_space,
                                            n_sampling) if n_sampling != -1 else -1
        self.auto_est.fit(
            data=data,
            epochs=epochs,
            validation_data=validation_data,
            metric=self.metric,
            metric_mode=self.metric_mode,
            metric_threshold=metric_threshold,
            n_sampling=n_sampling,
            search_space=self.search_space,
            search_alg=search_alg,
            search_alg_params=search_alg_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
        )
        self.best_model = self.auto_est._get_best_automl_model()
        self.best_config = self.auto_est.get_best_config()

    def predict(self, data, batch_size=32):
        '''
        Predict using a the trained model after HPO(Hyper Parameter Optimization).

        :param data: a numpy ndarray x, where x's shape is (num_samples, lookback, feature_dim)
               where lookback and feature_dim should be the same as past_seq_len and
               input_feature_num.
        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time). The value
               defaults to 32.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        '''
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.best_model is None:
            invalidInputError(False,
                              "You must call fit or load first before calling predict!")
        return self.best_model.predict(data, batch_size=batch_size)

    def predict_with_onnx(self, data, batch_size=32, dirname=None):
        '''
        Predict using a the trained model after HPO(Hyper Parameter Optimization).

        Be sure to install onnx and onnxruntime to enable this function. The method
        will give exactly the same result as .predict() but with higher throughput
        and lower latency. keras will support onnx later.

        :param data: a numpy ndarray x, where x's shape is (num_samples, lookback, feature_dim)
               where lookback and feature_dim should be the same as past_seq_len and
               input_feature_num.
        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time). The value
               defaults to 32.
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        '''
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.backend.startswith("keras"):
            invalidInputError(False,
                              "Currenctly, keras not support onnx method.")

        if self.best_model is None:
            invalidInputError(False,
                              "You must call fit or load first before calling predict!")
        return self.best_model.predict_with_onnx(data, batch_size=batch_size, dirname=dirname)

    def evaluate(self, data,
                 batch_size=32,
                 metrics=["mse"],
                 multioutput="raw_values"):
        '''
        Evaluate using a the trained model after HPO(Hyper Parameter Optimization).

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from bigdl.orca.automl.metrics import Evaluator
        >>> y_hat = automodel.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: a numpy ndarray tuple (x, y) x's shape is (num_samples, lookback,
               feature_dim) where lookback and feature_dim should be the same as
               past_seq_len and input_feature_num. y's shape is (num_samples, horizon,
               target_dim), where horizon and target_dim should be the same as
               future_seq_len and output_target_num.
        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param metrics: list of string or callable. e.g. ['mse'] or [customized_metrics]
               If callable function, it signature should be func(y_true, y_pred), where y_true and
               y_pred are numpy ndarray. The function should return a float value as evaluation
               result.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.

        :return: A list of evaluation results. Each item represents a metric.
        '''
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.best_model is None:
            invalidInputError(False,
                              "You must call fit or load first before calling predict!")
        return self.best_model.evaluate(data[0], data[1], metrics=metrics,
                                        multioutput=multioutput, batch_size=batch_size)

    def evaluate_with_onnx(self, data,
                           batch_size=32,
                           metrics=["mse"],
                           dirname=None,
                           multioutput="raw_values"):
        '''
        Evaluate using a the trained model after HPO(Hyper Parameter Optimization).

        Be sure to install onnx and onnxruntime to enable this function. The method
        will give exactly the same result as .evaluate() but with higher throughput
        and lower latency. keras will support onnx later.

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from bigdl.orca.automl.metrics import Evaluator
        >>> y_hat = automodel.predict_with_onnx(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: a numpy ndarray tuple (x, y) x's shape is (num_samples, lookback,
               feature_dim) where lookback and feature_dim should be the same as
               past_seq_len and input_feature_num. y's shape is (num_samples, horizon,
               target_dim), where horizon and target_dim should be the same as
               future_seq_len and output_target_num.
        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param metrics: list of string or callable. e.g. ['mse'] or [customized_metrics]
               If callable function, it signature should be func(y_true, y_pred), where y_true and
               y_pred are numpy ndarray. The function should return a float value as evaluation
               result.
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.

        :return: A list of evaluation results. Each item represents a metric.
        '''
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.backend.startswith("keras"):
            invalidInputError(False,
                              "Currenctly, keras not support onnx method.")

        if self.best_model is None:
            invalidInputError(False,
                              "You must call fit or load first before calling predict!")
        return self.best_model.evaluate_with_onnx(data[0], data[1],
                                                  metrics=metrics,
                                                  dirname=dirname,
                                                  multioutput=multioutput,
                                                  batch_size=batch_size)

    def save(self, checkpoint_path):
        """
        Save the best model.

        Please note that if you only want the pytorch model or onnx model
        file, you can call .get_model() or .export_onnx_file(). The checkpoint
        file generated by .save() method can only be used by .load() in automodel.
        If you specify "keras" as backend, file name will be best_keras_config.json
        and best_keras_model.ckpt.

        :param checkpoint_path: The location you want to save the best model.
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.best_model is None:
            invalidInputError(False,
                              "You must call fit or load first before calling predict!")
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        model_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_MODEL_DIR)
        best_config_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_CONFIG_DIR)
        self.best_model.save(model_path)
        with open(best_config_path, "w") as f:
            json.dump(self.best_config, f)

    def load(self, checkpoint_path):
        """
        restore the best model.

        :param checkpoint_path: The checkpoint location you want to load the best model.
        """
        model_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_MODEL_DIR)
        best_config_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_CONFIG_DIR)
        self.best_model.restore(model_path)
        with open(best_config_path, "r") as f:
            self.best_config = json.load(f)

    def build_onnx(self, thread_num=1, sess_options=None):
        '''
        Build onnx model to speed up inference and reduce latency.
        The method is Not required to call before predict_with_onnx,
        evaluate_with_onnx or export_onnx_file.
        It is recommended to use when you want to:

        | 1. Strictly control the thread to be used during inferencing.
        | 2. Alleviate the cold start problem when you call predict_with_onnx
             for the first time.

        :param thread_num: int, the num of thread limit. The value is set to 1 by
               default where no limit is set. Besides, the environment variable
               `OMP_NUM_THREADS` is suggested to be same as `thread_num`.
        :param sess_options: an onnxruntime.SessionOptions instance, if you set this
               other than None, a new onnxruntime session will be built on this setting
               and ignore other settings you assigned(e.g. thread_num...).

        Example:
            >>> # to pre build onnx sess
            >>> automodel.build_onnx(thread_num=2)  # build onnx runtime sess for two threads
            >>> pred = automodel.predict_with_onnx(data)
            >>> # ------------------------------------------------------
            >>> # directly call onnx related method is also supported
            >>> # default to build onnx runtime sess for single thread
            >>> pred = automodel.predict_with_onnx(data)
        '''
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.backend.startswith("keras"):
            invalidInputError(False,
                              "Currenctly, keras not support onnx method.")

        import onnxruntime
        if sess_options is not None and not isinstance(sess_options, onnxruntime.SessionOptions):
            invalidInputError(False,
                              "sess_options should be an onnxruntime.SessionOptions instance"
                              f", but found {type(sess_options)}")
        if self.distributed:
            invalidInputError(False,
                              "build_onnx has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        try:
            OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
        except KeyError:
            OMP_NUM_THREADS = 0
        if OMP_NUM_THREADS != str(thread_num):
            warnings.warn("The environment variable OMP_NUM_THREADS is suggested to be same "
                          f"as thread_num.You can use 'export OMP_NUM_THREADS={thread_num}'.")

        import torch
        dummy_input = torch.rand(1, self.best_config["past_seq_len"],
                                 self.best_config["input_feature_num"])
        self.best_model._build_onnx(dummy_input,
                                    dirname=None,
                                    thread_num=thread_num,
                                    sess_options=None)

    def export_onnx_file(self, dirname):
        """
        Save the onnx model file to the disk.

        :param dirname: The dir location you want to save the onnx file.
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.backend.startswith("keras"):
            invalidInputError(False,
                              "Currenctly, keras not support onnx method.")

        if self.distributed:
            invalidInputError(False,
                              "export_onnx_file has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        import torch
        dummy_input = torch.rand(1, self.best_config["past_seq_len"],
                                 self.best_config["input_feature_num"])
        self.best_model._build_onnx(dummy_input, dirname)

    def get_best_model(self):
        """
        Get the best pytorch model.
        """
        return self.auto_est.get_best_model()

    def get_best_config(self):
        """
        Get the best configuration

        :return: A dictionary of best hyper parameters
        """
        return self.best_config

    def _get_best_automl_model(self):
        return self.best_model

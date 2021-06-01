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

from zoo.chronos.model.Seq2Seq_pytorch import Seq2SeqPytorch
from zoo.chronos.model.forecast.abstract import Forecaster
from zoo.automl.common.util import load_config

import os


class Seq2SeqForecaster(Forecaster):
    """
        Example:
            >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
            >>> forecaster = Seq2SeqForecaster(future_seq_len=5,
                                       input_feature_num=3,
                                       output_feature_num=2,
                                       lstm_layer_num=2)
            >>> train_mse = forecaster.fit(x_train, x_val, epochs=10)
            >>> test_pred = forecaster.predict(x_test)
            >>> test_mse = forecaster.evaluate(x_test, y_test)
            >>> forecaster.save({ckpt_name})
            >>> forecaster.restore({ckpt_name})
    """
    def __init__(self,
                 input_feature_num,
                 future_seq_len,
                 output_feature_num,
                 lstm_hidden_dim=128,
                 lstm_layer_num=1,
                 teacher_forcing=False,
                 dropout=0.25,
                 lr=0.001,
                 loss="mse",
                 optimizer="Adam",
                 ):
        """
        Build a LSTM Sequence to Sequence Forecast Model.

        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param input_feature_num: Specify the feature dimension.
        :param output_feature_num: Specify the output dimension.
        :param lstm_hidden_dim: LSTM hidden channel for decoder and encoder.
        :param lstm_layer_num: LSTM layer number for decoder and encoder.
        :param teacher_forcing: If use teacher forcing in training.
        :param dropout: Specify the dropout close possibility (i.e. the close
               possibility to a neuron). This value defaults to 0.25.
        :param optimizer: Specify the optimizer used for training. This value
               defaults to "Adam".
        :param loss: Specify the loss function used for training. This value
               defaults to "mse". You can choose from "mse", "mae" and
               "huber_loss".
        :param lr: Specify the learning rate. This value defaults to 0.001.
        """
        self.check_optional_config = False
        self.model_config = {
            "input_feature_num": input_feature_num,
            "future_seq_len": future_seq_len,
            "output_feature_num": output_feature_num,
            "lstm_hidden_dim": lstm_hidden_dim,
            "lstm_layer_num": lstm_layer_num,
            "teacher_forcing": teacher_forcing,
            "dropout": dropout,
            "lr": lr,
            "loss": loss,
            "optimizer": optimizer,
        }
        self.internal = Seq2SeqPytorch(check_optional_config=False)

    def _check_data(self, x, y):
        assert self.model_config["future_seq_len"] == y.shape[-2], \
            "The y shape should be (batch_size, future_seq_len, target_col_num), \
            Got future_seq_len of {} in config while y input shape of {}."\
            .format(self.model_config["future_seq_len"], y.shape[-2])
        assert self.model_config["input_feature_num"] == x.shape[-1],\
            "The x shape should be (batch_size, past_seq_len, input_feature_num), \
            Got input_feature_num of {} in config while x input shape of {}."\
            .format(self.model_config["input_feature_num"], x.shape[-1])
        assert self.model_config["output_feature_num"] == y.shape[-1], \
            "The y shape should be (batch_size, future_seq_len, output_feature_num), \
            Got output_feature_num of {} in config while y input shape of {}."\
            .format(self.model_config["output_feature_num"], y.shape[-1])

    def fit(self, x, y, validation_data=None, epochs=1, metric="mse", batch_size=32):
        """
        Fit(Train) the forecaster.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
               lookback and feature_dim should be the same as past_seq_len and input_feature_num.
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
               horizon and target_dim should be the same as future_seq_len and output_feature_num.
        :param validation_data: A tuple (x_valid, y_valid) as validation data. Default to None.
        :param epochs: Number of epochs you want to train.
        :param metric: The metric for training data.
        :param batch_size: Number of batch size you want to train.

        :return: Evaluation results on validation data.
        """
        if validation_data is None:
            validation_data = (x, y)
        self.model_config["batch_size"] = batch_size
        self._check_data(x, y)
        return self.internal.fit_eval(data=(x, y),
                                      validation_data=validation_data,
                                      epochs=epochs,
                                      metric=metric,
                                      **self.model_config)

    def predict(self, x):
        """
        Predict using a trained forecaster.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).

        :return: A numpy array with shape (num_samples, lookback, feature_dim).
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict(x)

    def predict_with_onnx(self, x, dirname=None):
        """
        Predict using a trained forecaster with onnxruntime.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.

        :return: A numpy array with shape (num_samples, lookback, feature_dim).
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict_with_onnx(x, dirname=dirname)

    def evaluate(self, x, y, metrics=['mse'], multioutput="raw_values"):
        """
        Evaluate using a trained forecaster.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
        :param metrics: A list contains metrics for test/valid data.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.

        :return: A list of evaluation results. Each item represents a metric.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate(x, y, metrics=metrics, multioutput=multioutput)

    def evaluate_with_onnx(self, x, y,
                           metrics=['mse'],
                           dirname=None,
                           multioutput="raw_values"):
        """
        Evaluate using a trained forecaster with onnxruntime.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
        :param metrics: A list contains metrics for test/valid data.
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.

        :return: A list of evaluation results. Each item represents a metric.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate_with_onnx(x, y,
                                                metrics=metrics,
                                                dirname=dirname,
                                                multioutput=multioutput)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling save!")
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.internal.restore(checkpoint_file)

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

from bigdl.chronos.model.tf1.MTNet_keras import MTNetKeras as MTNetKerasModel
from bigdl.chronos.model.tf2.MTNet_keras import MTNetKeras as MTNetKerasModel
from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.metric.forecast_metrics import Evaluator


class MTNetForecaster(Forecaster):
    """
        Example:
            >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
            >>> model = MTNetForecaster(target_dim=1,
                                feature_dim=x_train.shape[-1],
                                long_series_num=6,
                                series_length=2
                                )
            >>> x_train_long, x_train_short = model.preprocess_input(x_train)
            >>> x_val_long, x_val_short = model.preprocess_input(x_val)
            >>> x_test_long, x_test_short = model.preprocess_input(x_test)
            >>> model.fit([x_train_long, x_train_short],
                  y_train,
                  validation_data=([x_val_long, x_val_short], y_val),
                  batch_size=32,
                  distributed=False)
            >>> predict_result = [x_test_long, x_test_short]
    """

    def __init__(self,
                 target_dim=1,
                 feature_dim=1,
                 long_series_num=1,
                 series_length=1,
                 ar_window_size=1,
                 cnn_height=1,
                 cnn_hid_size=32,
                 rnn_hid_sizes=[16, 32],
                 lr=0.001,
                 loss="mae",
                 cnn_dropout=0.2,
                 rnn_dropout=0.2,
                 metric="mean_squared_error",
                 uncertainty: bool = False,
                 ):
        """
        Build a MTNet Forecast Model.

        :param target_dim: the dimension of model output
        :param feature_dim: the dimension of input feature
        :param long_series_num: the number of series for the long-term memory series
        :param series_length: the series size for long-term and short-term memory series
        :param ar_window_size: the auto regression window size in MTNet
        :param cnn_hid_size: the hidden layer unit for cnn in encoder
        :param rnn_hid_sizes: the hidden layers unit for rnn in encoder
        :param cnn_height: cnn filter height in MTNet
        :param metric: the metric for validation and evaluation
        :param uncertainty: whether to enable calculation of uncertainty
        :param lr: learning rate
        :param loss: the target function you want to optimize on
        :param cnn_dropout: the dropout possibility for cnn in encoder
        :param rnn_dropout: the dropout possibility for rnn in encoder
        """
        self.check_optional_config = False
        self.mc = uncertainty
        self.model_config = {
            "feature_num": feature_dim,
            "output_dim": target_dim,
            "metrics": [metric],
            "mc": uncertainty,
            "time_step": series_length,
            "long_num": long_series_num,
            "ar_window": ar_window_size,
            "cnn_height": cnn_height,
            "past_seq_len": (long_series_num + 1) * series_length,
            "cnn_hid_size": cnn_hid_size,
            "rnn_hid_sizes": rnn_hid_sizes,
            "lr": lr,
            "cnn_dropout": cnn_dropout,
            "rnn_dropout": rnn_dropout,
            "loss": loss
        }
        super().__init__()

        self.internal = MTNetKerasModel(check_optional_config=self.check_optional_config)

    def fit(self, data, epochs=2, batch_size=32, validation_data=None):
        self.model_config.update({'batch_size': batch_size})
        self.internal.fit_eval(data, validation_data=validation_data, epochs=epochs, verbose=1, **self.model_config)

    def predict(self, data):
        """
        :param data: x's a numpy dataset.
        """
        yhat = self.internal.predict(data)
        return yhat

    def evaluate(self, data, metrics=["mae"], multioutput='raw_values'):
        yhat = self.internal.predict(data[0])
        return [Evaluator.evaluate(m, y_true=data[1],
                                   y_pred=yhat, aggregate=multioutput) for m in metrics]

    def save(self, checkpoint):
        self.internal.save(checkpoint_file=checkpoint)

    def load(self, checkpoint):
        self.internal.restore(checkpoint, self.model_config)

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

from zoo.chronos.model.VanillaLSTM import VanillaLSTM as LSTMKerasModel
from zoo.chronos.model.forecast.tfpark_forecaster import TFParkForecaster


class LSTMForecaster(TFParkForecaster):
    """
    Example:
        >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
        >>> model = LSTMForecaster(target_dim=1, feature_dim=x_train.shape[-1])
        >>> model.fit(x_train,
                  y_train,
                  validation_data=(x_val, y_val),
                  batch_size=8,
                  distributed=False)
        >>> predict_result = model.predict(x_test)
    """

    def __init__(self,
                 target_dim=1,
                 feature_dim=1,
                 lstm_units=(16, 8),
                 dropouts=0.2,
                 metric="mean_squared_error",
                 lr=0.001,
                 loss="mse",
                 optimizer="Adam",
                 ):
        """
        Build a LSTM Forecast Model.

        :param target_dim: dimension of model output
        :param feature_dim: dimension of input feature
        :param lstm_units: num of units for LSTM layers.
            Positive int or a list/tuple of positive ints.
        :param dropouts: dropout for the dropout layers. The same dropout rate will be set to all
            layers if dropouts is a float while lstm_units has multiple elements.
        :param metric: the metric for validation and evaluation. For regression, we support
            Mean Squared Error: ("mean_squared_error", "MSE" or "mse"),
            Mean Absolute Error: ("mean_absolute_error","MAE" or "mae"),
            Mean Absolute Percentage Error: ("mean_absolute_percentage_error", "MAPE", "mape")
            Cosine Proximity: ("cosine_proximity", "cosine")
        :param lr: learning rate
        :param loss: the target function you want to optimize on. Defaults to mse.
        :param optimizer: the optimizer used for training. Defaults to Adam.
        """
        self.check_optional_config = False

        self.model_config = {
            "input_dim": feature_dim,
            "output_dim": target_dim,
            "lr": lr,
            "lstm_units": lstm_units,
            "dropouts": dropouts,
            "optim": optimizer,
            "metric": metric,
            "loss": loss,
        }
        self.internal = None

        super().__init__()

    def _build(self):
        """
        Build LSTM Model in tf.keras
        """
        # build model with TF/Keras
        self.internal = LSTMKerasModel(check_optional_config=self.check_optional_config)
        return self.internal.model_creator(self.model_config)

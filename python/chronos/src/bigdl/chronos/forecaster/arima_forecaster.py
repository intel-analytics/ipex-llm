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
from bigdl.chronos.model.arima import ARIMAModel


class ARIMAForecaster(Forecaster):
    """
    Example:
        >>> #The dataset is split into data, validation_data
        >>> model = ARIMAForecaster(p=2, q=2, seasonality_mode=False)
        >>> model.fit(data, validation_data)
        >>> predict_result = model.predict(horizon=24)
    """

    def __init__(self,
                 p=2,
                 q=2,
                 seasonality_mode=True,
                 P=3,
                 Q=1,
                 m=7,
                 metric="mse",
                 ):
        """
        Build a ARIMA Forecast Model.
        User can customize p, q, seasonality_mode, P, Q, m, metric for the ARIMA model,
        the differencing term (d) and seasonal differencing term (D) are automatically
        estimated from the data. For details of the ARIMA model hyperparameters, refer to
        https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.

        :param p: hyperparameter p for the ARIMA model.
        :param q: hyperparameter q for the ARIMA model.
        :param seasonality_mode: hyperparameter seasonality_mode for the ARIMA model.
        :param P: hyperparameter P for the ARIMA model.
        :param Q: hyperparameter Q for the ARIMA model.
        :param m: hyperparameter m for the ARIMA model.
        :param metric: the metric for validation and evaluation. For regression, we support
            Mean Squared Error: ("mean_squared_error", "MSE" or "mse"),
            Mean Absolute Error: ("mean_absolute_error","MAE" or "mae"),
            Mean Absolute Percentage Error: ("mean_absolute_percentage_error", "MAPE", "mape")
            Cosine Proximity: ("cosine_proximity", "cosine")
        """
        self.model_config = {
            "p": p,
            "q": q,
            "seasonality_mode": seasonality_mode,
            "P": P,
            "Q": Q,
            "m": m,
            "metric": metric,
        }
        self.internal = ARIMAModel()

        super().__init__()

    def fit(self, data, validation_data):
        """
        Fit(Train) the forecaster.

        :param data: A 1-D numpy array as the training data
        :param validation_data: A 1-D numpy array as the evaluation data
        """
        self._check_data(data, validation_data)
        data = data.reshape(-1, 1)
        # validation_data = validation_data.reshape(-1, 1)
        return self.internal.fit_eval(data=data,
                                      validation_data=validation_data,
                                      **self.model_config)

    def _check_data(self, data, validation_data):
        assert data.ndim == 1, \
            "data should be an 1-D array), \
            Got data dimension of {}."\
            .format(data.ndim)
        assert validation_data.ndim == 1, \
            "validation_data should be an 1-D array), \
            Got validation_data dimension of {}."\
            .format(validation_data.ndim)

    def predict(self, horizon, rolling=False):
        """
        Predict using a trained forecaster.

        :param horizon: the number of steps forward to predict
        :param rolling: whether to use rolling prediction
        """
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict(horizon=horizon, rolling=rolling)

    def evaluate(self, validation_data, metrics=['mse'], rolling=False):
        """
        Evaluate using a trained forecaster.

        :param validation_data: A 1-D numpy array as the evaluation data
        :param metrics: A list contains metrics for test/valid data.
        """
        if validation_data is None:
            raise ValueError("Input invalid validation_data of None")
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate(validation_data, metrics=metrics, rolling=rolling)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling save!")
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        Restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.internal.restore(checkpoint_file)

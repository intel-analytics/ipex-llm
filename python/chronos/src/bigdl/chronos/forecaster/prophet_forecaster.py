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
from bigdl.chronos.model.prophet import ProphetModel


class ProphetForecaster(Forecaster):
    """
    Example:
        >>> #The dataset is split into data, validation_data
        >>> model = ProphetForecaster(changepoint_prior_scale=0.05, seasonality_mode='additive')
        >>> model.fit(data, validation_data)
        >>> predict_result = model.predict(horizon=24)
    """

    def __init__(self,
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0,
                 seasonality_mode='additive',
                 changepoint_range=0.8,
                 metric="mse",
                 ):
        """
        Build a Prophet Forecast Model.
        User can customize changepoint_prior_scale, seasonality_prior_scale,
        holidays_prior_scale, seasonality_mode, changepoint_range and metric
        of the Prophet model, for details of the Prophet model hyperparameters, refer to
        https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning.

        :param changepoint_prior_scale: hyperparameter changepoint_prior_scale for the
            Prophet model.
        :param seasonality_prior_scale: hyperparameter seasonality_prior_scale for the
            Prophet model.
        :param holidays_prior_scale: hyperparameter holidays_prior_scale for the
            Prophet model.
        :param seasonality_mode: hyperparameter seasonality_mode for the
            Prophet model.
        :param changepoint_range: hyperparameter changepoint_range for the
            Prophet model.
        :param metric: the metric for validation and evaluation. For regression, we support
            Mean Squared Error: ("mean_squared_error", "MSE" or "mse"),
            Mean Absolute Error: ("mean_absolute_error","MAE" or "mae"),
            Mean Absolute Percentage Error: ("mean_absolute_percentage_error", "MAPE", "mape")
            Cosine Proximity: ("cosine_proximity", "cosine")
        """
        self.model_config = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": seasonality_mode,
            "changepoint_range": changepoint_range,
            "metric": metric
        }
        self.internal = ProphetModel()

        super().__init__()

    def fit(self, data, validation_data=None):
        """
        Fit(Train) the forecaster.

        :param data: training data, a pandas dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating value
            and Td is the time dimension
        :param validation_data: evaluation data, should be the same type as data

        :return: the evaluation metric value
        """
        self._check_data(data, validation_data)
        return self.internal.fit_eval(data=data,
                                      validation_data=validation_data,
                                      **self.model_config)

    def _check_data(self, data, validation_data):
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError('ds' in data.columns and 'y' in data.columns,
                          "data should be a pandas dataframe that has at"
                          " least 2 columns 'ds' and 'y'.")
        if validation_data is not None:
            invalidInputError('ds' in validation_data.columns and 'y' in validation_data.columns,
                              "validation_data should be a dataframe that has at least"
                              " 2 columns 'ds' and 'y'.")

    def predict(self, horizon=1, freq="D", ds_data=None):
        """
        Predict using a trained forecaster.

        :param horizon: the number of steps forward to predict, the value defaults to 1.
        :param freq: the freqency of the predicted dataframe, defaulted to day("D"),
               the frequency can be anything from the pandas list of frequency strings here:
               https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param ds_data: a dataframe that has 1 column 'ds' indicating date.

        :return: A pandas DataFrame of length horizon,
                 including "trend" and "seasonality" and inference values, etc.
                 where the "yhat" column is the inference value.
        """
        if self.internal.model is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "You must call fit or restore first before calling predict!")
        return self.internal.predict(horizon=horizon, freq=freq, ds_data=ds_data)

    def evaluate(self, data, metrics=['mse']):
        """
        Evaluate using a trained forecaster.

        :param data: evaluation data, a pandas dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating value
            and Td is the time dimension
        :param metrics: A list contains metrics for test/valid data.

        :return: A list of evaluation results. Calculation results for each metrics.
        """
        from bigdl.nano.utils.common import invalidInputError
        if data is None:
            invalidInputError(False, "Input invalid data of None")
        if self.internal.model is None:
            invalidInputError(False,
                              "You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate(target=data,
                                      metrics=metrics)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster, should be a json file
        """
        if self.internal.model is None:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "You must call fit or restore first before calling save!")
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        Restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.internal.restore(checkpoint_file)

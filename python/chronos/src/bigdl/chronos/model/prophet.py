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

import json
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation

from bigdl.chronos.metric.forecast_metrics import Evaluator


class ProphetModel:

    def __init__(self):
        """
        Initialize Model
        """
        self.metric = 'mse'
        self.model = None
        self.model_init = False

    def _build(self, **config):
        """
        build the model and initialize.
        :param config: hyperparameters for the model
        """
        changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        holidays_prior_scale = config.get('holidays_prior_scale', 10.0)
        seasonality_mode = config.get('seasonality_mode', 'additive')
        changepoint_range = config.get('changepoint_range', 0.8)
        self.metric = config.get('metric', self.metric)
        self.metric_func = config.get('metric_func', None)

        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                             seasonality_prior_scale=seasonality_prior_scale,
                             holidays_prior_scale=holidays_prior_scale,
                             changepoint_range=changepoint_range,
                             seasonality_mode=seasonality_mode)

    def _fit(self, data, **config):
        """
        Only fit the data without validation data

        :param data: training data, an dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating target
            and Td is the time dimension
        """
        if not self.model_init:
            self._build(**config)
            self.model_init = True

        self.model.fit(data)

    def fit_eval(self,
                 data,
                 validation_data=None,
                 **config):
        """
        Fit on the training data from scratch.

        :param data: training data, an dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating target
            and Td is the time dimension
        :param validation_data: validation data, should be the same type as data.
        :param cross_validation: bool, if the eval result comes from cross_validation.
               The value is set to False by default. Setting this option to true may
               significantly slow down the process.
        :param expected_horizon: int or pandas interval str, only effective when
               cross_validation is set to True.
        :return: the evaluation metric value
        """
        self._fit(data, **config)

        # cross_validation
        cross_validation = config.get('cross_validation', False)
        expected_horizon = config.get('expect_horizon', int(0.1*len(data)))
        if cross_validation:
            return self._eval_cross_validation(expected_horizon)

        # normal validation process
        if validation_data is not None:
            if self.metric_func:
                val_metric = self.evaluate(target=validation_data,
                                           metrics=[self.metric_func])[0].item()
            else:
                val_metric = self.evaluate(target=validation_data,
                                           metrics=[self.metric])[0].item()
            if self.metric_func:
                return {self.metric_func.__name__: val_metric}
            else:
                return {self.metric: val_metric}

    def _eval_cross_validation(self, expected_horizon):
        df_cv = cross_validation(self.model, horizon=expected_horizon)
        df_p = performance_metrics(df_cv, metrics=[self.metric], rolling_window=1)
        return {self.metric: np.mean(df_p[self.metric].values)}  # here we use the mean metrics

    def predict(self, ds_data=None, horizon=24, freq="D"):
        """
        Predict horizon time-points ahead the input data in fit_eval
        :param data: Prophet predicts the horizon steps foreward from the training data.
            So data should be None as it is not used.
        :param horizon: horizon length to predict
        :param freq: the freqency of the predicted dataframe, defaulted to day("D"),
               the frequency can be anything from the pandas list of frequency strings.

        :return: predicted result of length horizon
        """
        from bigdl.nano.utils.common import invalidInputError
        if self.model is None:
            invalidInputError(False,
                              "Needs to call fit_eval or restore first before calling predict")
        if ds_data is not None:
            return self.model.predict(ds_data)
        future = self.model.make_future_dataframe(periods=horizon, freq=freq)
        out = self.model.predict(future)[-horizon:]
        return out

    def evaluate(self, target, data=None, metrics=['mse']):
        """
        Evaluate on the prediction results. We predict horizon time-points ahead the input data
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        target.
        :param data: Prophet predicts the horizon steps foreward from the training data.
            So data should be None as it is not used.
        :param target: target for evaluation. A dataframe with 2 columns, where column 'ds'
               indicating date and column 'y' indicating target.
        :param metrics: a list of metrics in string format or callable function with format
               it signature should be func(y_true, y_pred), where y_true and y_pred are numpy
               ndarray. The function should return a float value as evaluation result.

        :return: a list of metric evaluation results
        """
        from bigdl.nano.utils.common import invalidInputError
        if data is not None:
            invalidInputError(False,
                              "We don't support input data currently")
        if target is None:
            invalidInputError(False,
                              "Input invalid target of None")
        if self.model is None:
            invalidInputError(False,
                              "Needs to call fit_eval or restore first before calling evaluate")

        target_pred = self.model.predict(target)
        return Evaluator.evaluate(metrics, target.y.values,
                                  target_pred.yhat.values, aggregate="mean")

    def save(self, checkpoint):
        from bigdl.nano.utils.common import invalidInputError
        if self.model is None:
            invalidInputError(False,
                              "Needs to call fit_eval or restore first before calling save")
        with open(checkpoint, 'w') as fout:
            json.dump(model_to_json(self.model), fout)

    def restore(self, checkpoint):
        with open(checkpoint, 'r') as fin:
            self.model = model_from_json(json.load(fin))
        self.model_init = True


class ProphetBuilder:

    def __init__(self, **prophet_config):
        """
        Initialize Prophet Model
        :param prophet_config: Other prophet hyperparameters. You may refer to
           https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        for the parameter names to specify.
        """
        self.model_config = prophet_config.copy()

    def build(self, config):
        """
        Build Prophet Model
        :param config: Other prophet hyperparameters. You may refer to
           https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        for the parameter names to specify.
        """
        from bigdl.chronos.model.prophet import ProphetModel
        model = ProphetModel()
        model._build(**config)
        return model

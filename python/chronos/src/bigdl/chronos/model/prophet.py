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

import json
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from zoo.automl.common.metrics import Evaluator
from zoo.automl.model.abstract import BaseModel, ModelBuilder


class ProphetModel(BaseModel):

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

        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                             seasonality_prior_scale=seasonality_prior_scale,
                             holidays_prior_scale=holidays_prior_scale,
                             changepoint_range=changepoint_range,
                             seasonality_mode=seasonality_mode)

    def fit_eval(self, data, validation_data, **config):
        """
        Fit on the training data from scratch.

        :param data: training data, an dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating target
            and Td is the time dimension
        :param validation_data: validation data, should be the same type as data.
        :return: the evaluation metric value
        """
        if not self.model_init:
            self._build(**config)
            self.model_init = True

        self.model.fit(data)
        val_metric = self.evaluate(target=validation_data,
                                   metrics=[self.metric])[0].item()
        return {self.metric: val_metric}

    def predict(self, data=None, horizon=24):
        """
        Predict horizon time-points ahead the input data in fit_eval
        :param data: Prophet predicts the horizon steps foreward from the training data.
            So data should be None as it is not used.
        :param horizon: horizon length to predict
        :return: predicted result of length horizon
        """
        if data is not None:
            raise ValueError("We don't support input data currently")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        future = self.model.make_future_dataframe(periods=horizon)
        out = self.model.predict(future)[-horizon:]

        return out

    def evaluate(self, target, data=None, metrics=['mse']):
        """
        Evaluate on the prediction results. We predict horizon time-points ahead the input data
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        target.
        :param data: Prophet predicts the horizon steps foreward from the training data.
            So data should be None as it is not used.
        :param target: target for evaluation.
        :param metrics: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        if data is not None:
            raise ValueError("We don't support input data currently")
        if target is None:
            raise ValueError("Input invalid target of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling evaluate")

        horizon = len(target)
        future = self.model.make_future_dataframe(periods=horizon)
        target_pred = self.predict(horizon=horizon)[['yhat']]
        return [Evaluator.evaluate(m, target[['y']].values, target_pred.values) for m in metrics]

    def save(self, checkpoint):
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling save")
        with open(checkpoint, 'w') as fout:
            json.dump(model_to_json(self.model), fout)

    def restore(self, checkpoint):
        with open(checkpoint, 'r') as fin:
            self.model = model_from_json(json.load(fin))
        self.model_init = True


class ProphetBuilder(ModelBuilder):

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
        from zoo.chronos.model.prophet import ProphetModel
        model = ProphetModel()
        model._build(**config)
        return model

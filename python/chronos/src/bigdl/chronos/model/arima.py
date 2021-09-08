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

import os
import pickle
from pmdarima.arima import ARIMA
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs

from zoo.orca.automl.metrics import Evaluator
from zoo.orca.automl.model.abstract import BaseModel, ModelBuilder


class ARIMAModel(BaseModel):

    def __init__(self):
        """
        Initialize Model
        """
        self.seasonal = True
        self.metric = 'mse'
        self.model = None
        self.model_init = False

    def _build(self, **config):
        """
        build the models and initialize.
        :param config: hyperparameters for the model
        """
        p = config.get('p', 2)
        d = config.get('d', 0)
        q = config.get('q', 2)
        self.seasonal = config.get('seasonality_mode', True)
        P = config.get('P', 1)
        D = config.get('D', 0)
        Q = config.get('Q', 1)
        m = config.get('m', 7)
        self.metric = config.get('metric', self.metric)

        order = (p, d, q)
        if not self.seasonal:
            seasonal_order = (0, 0, 0, 0)
        else:
            seasonal_order = (P, D, Q, m)

        self.model = ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)

    def fit_eval(self, data, validation_data, **config):
        """
        Fit on the training data from scratch.
        :param data: A 1-D numpy array as the training data
        :param validation_data: A 1-D numpy array as the evaluation data
        :return: the evaluation metric value
        """

        if not self.model_init:
            # Estimating differencing term (d) and seasonal differencing term (D)
            kpss_diffs = ndiffs(data, alpha=0.05, test='kpss', max_d=6)
            adf_diffs = ndiffs(data, alpha=0.05, test='adf', max_d=6)
            d = max(adf_diffs, kpss_diffs)
            D = 0 if not self.seasonal else nsdiffs(data, m=7, max_D=12)
            config.update(d=d, D=D)

            self._build(**config)
            self.model_init = True

        self.model.fit(data)
        val_metric = self.evaluate(x=None, target=validation_data, metrics=[self.metric])[0].item()
        return {self.metric: val_metric}

    def predict(self, x=None, horizon=24, update=False, rolling=False):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: ARIMA predicts the horizon steps foreward from the training data.
            So x should be None as it is not used.
        :param horizon: the number of steps forward to predict
        :param update: whether to update the original model
        :param rolling: whether to use rolling prediction
        :return: predicted result of length horizon
        """
        if x is not None:
            raise ValueError("x should be None")
        if update and not rolling:
            raise Exception("We don't support updating model without rolling prediction currently")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")

        if not update and not rolling:
            forecasts = self.model.predict(n_periods=horizon)
        elif rolling:
            if not update:
                self.save("tmp.pkl")

            forecasts = []
            for step in range(horizon):
                fc = self.model.predict(n_periods=1).item()
                forecasts.append(fc)

                # Updates the existing model with a small number of MLE steps for rolling prediction
                self.model.update(fc)

            if not update:
                self.restore("tmp.pkl")
                os.remove("tmp.pkl")

        return forecasts

    def evaluate(self, target, x=None, metrics=['mse'], rolling=False):
        """
        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        y.
        :param target: target for evaluation.
        :param x: ARIMA predicts the horizon steps foreward from the training data.
            So x should be None as it is not used.
        :param metrics: a list of metrics in string format
        :param rolling: whether to use rolling prediction
        :return: a list of metric evaluation results
        """
        if x is not None:
            raise ValueError("We don't support input x currently")
        if target is None:
            raise ValueError("Input invalid target of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling evaluate")

        forecasts = self.predict(horizon=len(target), rolling=rolling)

        return [Evaluator.evaluate(m, target, forecasts) for m in metrics]

    def save(self, checkpoint_file):
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling save")
        with open(checkpoint_file, 'wb') as fout:
            pickle.dump(self.model, fout)

    def restore(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as fin:
            self.model = pickle.load(fin)
        self.model_init = True


class ARIMABuilder(ModelBuilder):

    def __init__(self, **arima_config):
        """
        Initialize ARIMA Model Builder
        :param ARIMA_config: Other ARIMA hyperparameters. You may refer to
           https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        for the parameter names to specify.
        """
        self.model_config = arima_config.copy()

    def build(self, config):
        """
        Build ARIMA Model
        :param config: Other ARIMA hyperparameters. You may refer to
           https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        for the parameter names to specify.
        """
        from zoo.chronos.model.arima import ARIMAModel
        model = ARIMAModel()
        model._build(**config)
        return model

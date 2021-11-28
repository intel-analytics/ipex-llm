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

import torch
from functools import partial
from torchmetrics.functional import mean_squared_error, mean_absolute_error,\
    mean_absolute_percentage_error, r2_score, symmetric_mean_absolute_percentage_error


TORCHMETRICS_REGRESSION_MAP = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'rmse': partial(mean_squared_error, squared=False),
    'mape': mean_absolute_percentage_error,
    'smape': symmetric_mean_absolute_percentage_error,
    'r2': r2_score,
}


class Evaluator(object):
    """
    Evaluate metrics for y_true and y_pred.
    """

    @staticmethod
    def evaluate(metric, y_true, y_pred, multioutput='raw_values'):
        """
        Evaluate a specific metric for y_true and y_pred.

        :param metric: String in ['mae', 'mse', 'rmse', 'r2'
               , 'mape', 'smape']
        :param y_true: Array-like of shape = (n_samples, \*).
               Ground truth (correct) target values.
        :param y_pred: Array-like of shape = (n_samples, \*).
               Estimated target values.
        :param multioutput: String in ['raw_values', 'uniform_average']

        :return: Float or ndarray of floats.
                 A floating point value, or an
                 array of floating point values, one for each individual target.
        """
        metric = metric.lower()
        assert metric in TORCHMETRICS_REGRESSION_MAP.keys(),\
            f"metric should be one of {TORCHMETRICS_REGRESSION_MAP.keys()}, "\
            f"but get {metric}."
        assert multioutput in ['raw_values', 'uniform_average'],\
            "multioutput should be one of ['raw_values', 'uniform_average'], "\
            f"but get {multioutput}."
        assert y_true.shape == y_pred.shape,\
            "y_true and y_pred should have the same shape, "\
            f"but get {y_true.shape} and {y_pred.shape}."
        y_true, y_pred = torch.from_numpy(y_true), torch.from_numpy(y_pred)

        res = None
        if multioutput=="raw_values":
            res = torch.zeros(y_true.shape[1], y_true.shape[2])
            for i in range(y_true.shape[1]):
                for j in range(y_true.shape[2]):
                    res[i, j] = TORCHMETRICS_REGRESSION_MAP[metric](y_pred[:, i, j],
                                                                    y_true[:, i, j])
        else:
            res = TORCHMETRICS_REGRESSION_MAP[metric](y_pred, y_true)

        return res.numpy()

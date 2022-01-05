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
from numpy import ndarray
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


def _standard_input(metrics, y_true, y_pred):
    """
    Standardize input functions. Format metrics,
    check the ndim of y_pred and y_true,
    converting 1-3 dim y_true and y_pred to 2 dim.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(map(lambda x: x.lower(), metrics))
    assert all(metric in TORCHMETRICS_REGRESSION_MAP.keys() for metric in metrics),\
        f"metric should be one of {TORCHMETRICS_REGRESSION_MAP.keys()}, "\
        f"but get {metrics}."
    assert type(y_true) is type(y_pred) and isinstance(y_pred, ndarray),\
        "y_pred and y_true type must be numpy.ndarray, "\
        f"but found {type(y_pred)} and {type(y_true)}."
    y_true, y_pred = torch.from_numpy(y_true), torch.from_numpy(y_pred)

    assert y_true.shape == y_pred.shape,\
        "y_true and y_pred should have the same shape, "\
        f"but get {y_true.shape} and {y_pred.shape}."

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        original_shape = y_true.shape
    elif y_true.ndim == 3:
        original_shape = y_true.shape
        y_true = y_true.reshape(y_true.shape[0], y_true.shape[1]*y_true.shape[2])
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1]*y_pred.shape[2])
    else:
        original_shape = y_true.shape
    return metrics, y_true, y_pred, original_shape


class Evaluator(object):
    """
    Evaluate metrics for y_true and y_pred.
    """

    @staticmethod
    def evaluate(metrics, y_true, y_pred, aggregate='mean'):
        """
        Evaluate a specific metrics for y_true and y_pred.

        :param metrics: String or list in ['mae', 'mse', 'rmse', 'r2', 'mape', 'smape']
        :param y_true: Array-like of shape = (n_samples, \*). Ground truth (correct) target values.
        :param y_pred: Array-like of shape = (n_samples, \*). Estimated target values.
        :param aggregate: aggregation method. Currently, "mean" and None are supported,
               'mean' represents aggregating by mean, while None will return the element-wise
               result. The value defaults to 'mean'.

        :return: Float or ndarray of floats.
                 A floating point value, or an
                 array of floating point values, one for each individual target.
        """
        metrics, y_true, y_pred, original_shape = _standard_input(metrics, y_true, y_pred)

        res_list = []
        for metric in metrics:
            if len(original_shape) in [2, 3] and aggregate is None:
                res = torch.zeros(y_true.shape[-1])
                for i in range(y_true.shape[-1]):
                    res[i] = TORCHMETRICS_REGRESSION_MAP[metric](y_pred[..., i], y_true[..., i])
                res = res.reshape(original_shape[1:])
            else:
                res = TORCHMETRICS_REGRESSION_MAP[metric](y_pred, y_true)
            res_list.append(res.numpy())
        return res_list

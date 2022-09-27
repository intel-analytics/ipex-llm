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

from numpy import ndarray
import numpy as np
from bigdl.nano.utils.log4Error import invalidInputError
from timeit import repeat


EPSILON = 1e-10


def mae(y_label, y_predict):
    """
    Calculate the mean absolute error (MAE).

    .. math::

        \\text{MAE} = \\frac{1}{n}\\sum_{t=1}^n |y_t-\\hat{y_t}|

    :param y_label: Array-like of shape = (n_samples, \*).
           Ground truth (correct) target values.
    :param y_predict: Array-like of shape = (n_samples, \*).
           Estimated target values.
    :return: Ndarray of floats.
             An array of non-negative floating point values (the best value is 0.0).
    """
    result = np.mean(np.abs(y_label - y_predict))
    return result


def mse(y_label, y_predict):
    """
    Calculate the mean squared error (MSE).

    .. math::

        \\text{MSE} = \\frac{1}{n}\\sum_{t=1}^n (y_t-\\hat{y_t})^2

    :param y_label: Array-like of shape = (n_samples, \*).
           Ground truth (correct) target values.
    :param y_predict: Array-like of shape = (n_samples, \*).
           Estimated target values.
    :return: Ndarray of floats.
             An array of non-negative floating point values (the best value is 0.0).
    """
    result = np.mean((y_label - y_predict) ** 2)
    return result


def rmse(y_label, y_predict):
    """
    Calculate square root of the mean squared error (RMSE).

    .. math::

        \\text{RMSE} = \\sqrt{(\\frac{1}{n}\\sum_{t=1}^n (y_t-\\hat{y_t})^2)}

    :param y_label: Array-like of shape = (n_samples, \*).
           Ground truth (correct) target values.
    :param y_predict: Array-like of shape = (n_samples, \*).
           Estimated target values.
    :return: Ndarray of floats.
             An array of non-negative floating point values (the best value is 0.0).
    """
    return np.sqrt(mse(y_label, y_predict))


def mape(y_label, y_predict):
    """
    Calculate mean absolute percentage error (MAPE).

    .. math::

        \\text{MAPE} = \\frac{100\%}{n}\\sum_{t=1}^n  |\\frac{y_t-\\hat{y_t}}{y_t}|

    :param y_label: Array-like of shape = (n_samples, \*).
           Ground truth (correct) target values.
    :param y_predict: Array-like of shape = (n_samples, \*).
           Estimated target values.
    :return: Ndarray of floats.
             An array of non-negative floating point values (the best value is 0.0).
    """
    return np.mean(np.abs((y_label - y_predict) / (y_label + EPSILON)))


def smape(y_label, y_predict):
    """
    Calculate Symmetric mean absolute percentage error (sMAPE).

    .. math::

        \\text{sMAPE} = \\frac{100\%}{n} \\sum_{t=1}^n \\frac{|y_t-\\hat{y_t}|}{|y_t|+|\\hat{y_t}|}

    :param y_label: Array-like of shape = (n_samples, \*).
           Ground truth (correct) target values.
    :param y_predict: Array-like of shape = (n_samples, \*).
           Estimated target values.
    :return: Ndarray of floats.
             An array of non-negative floating point values (the best value is 0.0).
    """
    abs_diff = np.abs(y_predict - y_label)
    abs_per_error = abs_diff / (np.abs(y_predict) + np.abs(y_label) + EPSILON)
    sum_abs_per_error = np.mean(abs_per_error)
    return sum_abs_per_error * 100


def r2(y_label, y_predict):
    """
    Calculate the r2 score.

    .. math::

        R^2 = 1-\\frac{\\sum_{t=1}^n (y_t-\\hat{y_t})^2}{\\sum_{t=1}^n (y_t-\\bar{y})^2}

    :param y_label: Array-like of shape = (n_samples, \*).
           Ground truth (correct) target values.
    :param y_predict: Array-like of shape = (n_samples, \*).
           Estimated target values.
    :return: Ndarray of floats.
             An array of non-negative floating point values (the best value is 1.0).
    """
    return 1 - np.sum((y_label - y_predict)**2) / np.sum((y_label - np.mean(y_label))**2)


REGRESSION_MAP = {
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'mape': mape,
    'smape': smape,
    'r2': r2,
}


def _standard_input(metrics, y_true, y_pred):
    """
    Standardize input functions. Format metrics,
    check the ndim of y_pred and y_true,
    converting 1-3 dim y_true and y_pred to 2 dim.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    if isinstance(metrics[0], str):
        metrics = list(map(lambda x: x.lower(), metrics))
        invalidInputError(all(metric in REGRESSION_MAP.keys() for metric in metrics),
                          f"metric should be one of {REGRESSION_MAP.keys()},"
                          f" but get {metrics}.")
        invalidInputError(type(y_true) is type(y_pred) and isinstance(y_pred, ndarray),
                          "y_pred and y_true type must be numpy.ndarray,"
                          f" but found {type(y_pred)} and {type(y_true)}.")

    invalidInputError(y_true.shape == y_pred.shape,
                      "y_true and y_pred should have the same shape, "
                      f"but get {y_true.shape} and {y_pred.shape}.")

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
        :param metrics: String or list in ['mae', 'mse', 'rmse', 'r2', 'mape', 'smape'] for built-in
               metrics. If callable function, it signature should be func(y_true, y_pred), where
               y_true and y_pred are numpy ndarray.
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
            if callable(metric):
                metric_func = metric
            else:
                metric_func = REGRESSION_MAP[metric]
            if len(original_shape) in [2, 3] and aggregate is None:
                res = np.zeros(y_true.shape[-1])
                for i in range(y_true.shape[-1]):
                    res[i] = metric_func(y_true[..., i], y_pred[..., i])
                res = res.reshape(original_shape[1:])
                res_list.append(res)
            else:
                res = metric_func(y_true, y_pred)
                res_list.append(res)
        return res_list

    @staticmethod
    def get_latency(func, *args, num_running=100, **kwargs):
        """
        Return the time cost in milliseconds of a specific function by running multiple times.

        :param func: The function to be tested for latency.
        :param args: arguments for the tested function.
        :param num_running: Int and the value is positive. Specify the running number of
               the function and the value defaults to 100.
        :param kwargs: other arguments for the tested function.

        :return: Dictionary of str:float.
                 Show the information of the time cost in milliseconds.

        Example:
            >>> # to get the inferencing performance of a trained TCNForecaster
            >>> x = next(iter(test_loader))[0]
            >>> # run forecaster.predict(x.numpy()) for len(tsdata_test.df) times
            >>> # to evaluate the time cost
            >>> latency = Evaluator.get_latency(forecaster.predict, x.numpy(),\
num_running = len(tsdata_test.df))
            >>> # an example output:
            >>> # {"p50": 3.853, "p90": 3.881, "p95": 3.933, "p99": 4.107}
        """
        invalidInputError(isinstance(num_running, int), "num_running type must be int, "
                          f"but found {type(num_running)}.")
        if num_running < 0:
            invalidInputError(False, "num_running value must be positive, "
                              f"but found {num_running}.")

        time_list = repeat(lambda: func(*args, **kwargs), number=1, repeat=num_running)
        sorted_time = np.sort(time_list)

        latency_list = {"p50": round(1000 * np.median(time_list), 3),
                        "p90": round(1000 * sorted_time[int(0.90 * num_running)], 3),
                        "p95": round(1000 * sorted_time[int(0.95 * num_running)], 3),
                        "p99": round(1000 * sorted_time[int(0.99 * num_running)], 3)}

        return latency_list

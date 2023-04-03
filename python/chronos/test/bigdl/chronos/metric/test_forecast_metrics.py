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

import numpy as np
import pytest
import time
from unittest import TestCase
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from bigdl.chronos.metric.forecast_metrics import Evaluator
from .. import op_torch, op_tf2, op_distributed, op_diff_set_all

@op_torch
@op_tf2
class TestChronosForecastMetrics(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forecast_metric(self):
        n_samples = 50
        y_true = np.arange(n_samples) + 1
        y_pred = y_true + 1

        assert_almost_equal(Evaluator.evaluate("mse", y_true, y_pred, aggregate="mean")[0], 1.)
        assert_almost_equal(Evaluator.evaluate("mae", y_true, y_pred, aggregate="mean")[0], 1.)
        assert_almost_equal(Evaluator.evaluate("r2", y_true, y_pred, aggregate="mean")[0], 0.995, 2)
        assert_almost_equal(Evaluator.evaluate("smape", y_true, y_pred, aggregate="mean")[0], 3.895, 3)
        # 3-dim r2
        assert_almost_equal(Evaluator.evaluate("r2",
                                               y_true.reshape(5, 5, 2),
                                               y_pred.reshape(5, 5, 2),
                                               aggregate='mean')[0], 0.995, 2)
        # 2-dim r2
        assert_almost_equal(np.mean(Evaluator.evaluate("r2",
                                                       y_true.reshape(25, 2),
                                                       y_pred.reshape(25, 2),
                                                       aggregate=None)[0]), 0.995, 2)

        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, -0.3, 2, 8])

        assert_almost_equal(Evaluator.evaluate("mape", y_true, y_pred, aggregate="mean")[0], 17.74/100, 2)
        assert_almost_equal(Evaluator.evaluate("RMSE", y_true, y_pred, aggregate="mean")[0], 0.57, 2)

    def test_highdim_array_metrics(self):
        y_true = np.array([[[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]]])
        y_pred = np.array([[[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]]])

        # single metric
        # 3-dim
        assert_almost_equal(Evaluator.evaluate("smape", y_true, y_pred, aggregate=None)[0],
                            [[9.09, 25.], [0., 6.67]], 2)
        assert_almost_equal(Evaluator.evaluate("mape", y_true, y_pred, aggregate=None)[0],
                            [[16.67/100, 40.00/100], [0/100, 14.29/100]], 2)
        assert_almost_equal(Evaluator.evaluate("rmse", y_true, y_pred, aggregate=None)[0],
                            [[0.5, 0.2], [0, 1]], 2)
        assert_almost_equal(Evaluator.evaluate("mse", y_true, y_pred, aggregate=None)[0],
                            [[0.25, 0.04], [0, 1]], 2)

        # 2-dim
        y_true = np.array([[1, 2], [0.4, 5], [1, 2], [0.4, 5]])
        y_pred = np.array([[2, 1], [0.2, 3], [2, 1], [0.2, 3]])
        assert_almost_equal(Evaluator.evaluate("mse", y_true, y_pred, aggregate=None)[0],
                            [0.52, 2.5], 2)
        assert_almost_equal(Evaluator.evaluate('smape', y_true, y_pred, aggregate=None)[0],
                            [33.33, 29.17], 2)

        # multi metrics
        y_true = np.array([[[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]]])
        y_pred = np.array([[[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]]])
        mse, rmse, mape, smape = Evaluator.evaluate(['mse', 'rmse', 'mape', 'smape'],
                                                    y_true,
                                                    y_pred,
                                                    aggregate=None)
        assert_almost_equal(mse, [[0.25, 0.04], [0, 1]], 2)
        assert_almost_equal(rmse, [[0.5, 0.2], [0, 1]], 2)
        assert_almost_equal(mape, [[16.67/100, 40.00/100], [0/100, 14.29/100]], 2)
        assert_almost_equal(smape, [[9.09, 25.0], [0.0, 6.67]], 2)

    def test_standard_input(self):
        y_true = np.random.randn(100, 2, 2)
        y_pred = np.random.randn(100, 2, 2)

        with pytest.raises(RuntimeError):
            Evaluator.evaluate("test_smape", y_true, y_pred, aggregate=None)
        with pytest.raises(RuntimeError):
            Evaluator.evaluate("mse", y_true, y_pred.reshape(100, 4))
        y_true = [10, 2, 5]
        with pytest.raises(RuntimeError):
            Evaluator.evaluate('mse', y_true, y_true)

    @op_distributed
    def test_smape_equal_orca(self):
        from bigdl.orca.automl.metrics import sMAPE
        y_true = np.random.randn(100, 4)
        y_pred = np.random.randn(100, 4)
        smape = Evaluator.evaluate("smape", y_true, y_pred, aggregate="mean")[0]
        orca_smape = sMAPE(y_true, y_pred, multioutput='uniform_average')
        assert_almost_equal(smape, orca_smape, 6)
    
    def test_get_latency(self):
        def test_func(count):
            time.sleep(0.001*count)
        with pytest.raises(RuntimeError):
            Evaluator.get_latency(test_func, 5, num_running = "10")
        with pytest.raises(RuntimeError):
            Evaluator.get_latency(test_func, 5, num_running = -10)

        latency_list = Evaluator.get_latency(test_func, 5)
        assert isinstance(latency_list, dict)
        for info in ["p50", "p90", "p95", "p99"]:
            assert info in latency_list
            assert isinstance(latency_list[info], float)
    
    @op_diff_set_all
    def test_plot(self):
        # TODO: more tests
        y = np.random.randn(100, 24, 1)
        pred = np.random.randn(100, 24, 1)
        x = np.random.randn(100, 48, 1)
        std = np.random.randn(100, 24, 1)
        Evaluator.plot(pred, x=x, ground_truth=y, std=std,
                       prediction_interval=0.95, layout=(3,4), figsize=(16,8))

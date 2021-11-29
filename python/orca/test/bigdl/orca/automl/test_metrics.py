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
from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.orca.automl.metrics import ME, MAE, MSE, RMSE, MSLE, R2

from bigdl.orca.automl.metrics import MPE, MAPE, MSPE, sMAPE, MDAPE, sMDAPE
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal


class TestMetrics(ZooTestCase):

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_metrics(self, n_samples=50):
        y_true = np.arange(n_samples) + 1
        y_pred = y_true + 1

        assert_almost_equal(MSE(y_true, y_pred), [1.])
        assert_almost_equal(MSLE(y_true, y_pred),
                            MSE(np.log(1 + y_true),
                            np.log(1 + y_pred)))
        assert_almost_equal(MAE(y_true, y_pred), [1.])
        assert_almost_equal(R2(y_true, y_pred), [0.995], 2)

        assert_almost_equal(sMAPE(y_true, y_pred), [3.89], decimal=2)

        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, -0.3, 2, 8]

        assert_almost_equal(MAPE(y_true, y_pred), [17.74], decimal=2)
        assert_almost_equal(MPE(y_true, y_pred), [10.6], decimal=2)
        assert_almost_equal(RMSE(y_true, y_pred), [0.57], decimal=2)
        assert_almost_equal(ME(y_true, y_pred), [-0.17], decimal=2)
        assert_almost_equal(MSPE(y_true, y_pred), [5.2], decimal=2)
        assert_almost_equal(MDAPE(y_true, y_pred), [15.48], decimal=2)
        assert_almost_equal(sMDAPE(y_true, y_pred), [7.88], decimal=2)

    def test_multioutput_metrics(self):
        y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
        y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])
        assert_almost_equal(MSE(y_true, y_pred, multioutput='uniform_average'),
                            [(1. / 3 + 2. / 3 + 2. / 3) / 4.])

        assert_almost_equal(MSLE(y_true, y_pred, multioutput='uniform_average'),
                            [0.200], decimal=2)

        assert_almost_equal(MAE(y_true, y_pred, multioutput='uniform_average'),
                            [(1. + 2. / 3) / 4.])

        assert_almost_equal(R2(y_true, y_pred, multioutput='variance_weighted'), [1. - 5. / 2])

        assert_almost_equal(R2(y_true, y_pred, multioutput='uniform_average'), [-.875])

        y_true = ([[3, -0.5, 2, 7], [3, -0.5, 2, 7], [3, -0.5, 2, 7]])
        y_pred = ([[2.5, -0.3, 2, 8], [2.5, -0.3, 2, 8], [2.5, -0.3, 2, 8]])

        assert_almost_equal(sMAPE(y_true, y_pred, multioutput='uniform_average'),
                            [10.19], decimal=2)
        assert_almost_equal(MAPE(y_true, y_pred, multioutput='uniform_average'), [17.74], decimal=2)
        assert_almost_equal(MPE(y_true, y_pred, multioutput='uniform_average'), [10.6], decimal=2)
        assert_almost_equal(RMSE(y_true, y_pred, multioutput='uniform_average'), [0.57], decimal=2)

        assert_almost_equal(ME(y_true, y_pred, multioutput='uniform_average'), [-0.18], decimal=2)
        assert_almost_equal(MSPE(y_true, y_pred, multioutput='uniform_average'), [5.2], decimal=2)
        assert_almost_equal(MDAPE(y_true, y_pred, multioutput='uniform_average'),
                            [17.74], decimal=2)
        assert_almost_equal(sMDAPE(y_true, y_pred, multioutput='uniform_average'),
                            [10.19], decimal=2)

    def test_highdim_metrics(self):
        y_true = ([[[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]]])
        y_pred = ([[[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]]])
        assert_almost_equal(sMAPE(y_true, y_pred, multioutput='uniform_average'),
                            [10.19], decimal=2)
        assert_almost_equal(MAPE(y_true, y_pred, multioutput='uniform_average'), [17.74], decimal=2)
        assert_almost_equal(MPE(y_true, y_pred, multioutput='uniform_average'), [10.6], decimal=2)
        assert_almost_equal(RMSE(y_true, y_pred, multioutput='uniform_average'), [0.57], decimal=2)

        assert_almost_equal(ME(y_true, y_pred, multioutput='uniform_average'), [-0.18], decimal=2)
        assert_almost_equal(MSPE(y_true, y_pred, multioutput='uniform_average'), [5.2], decimal=2)
        assert_almost_equal(MDAPE(y_true, y_pred, multioutput='uniform_average'),
                            [17.74], decimal=2)
        assert_almost_equal(sMDAPE(y_true, y_pred, multioutput='uniform_average'),
                            [10.19], decimal=2)
        assert_almost_equal(MSE(y_true, y_pred, multioutput='uniform_average'), [0.32], decimal=2)

    def test_highdim_array_metrics(self):
        y_true = ([[[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]], [[3, -0.5], [2, 7]]])
        y_pred = ([[[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]], [[2.5, -0.3], [2, 8]]])

        assert_almost_equal(sMAPE(y_true, y_pred, multioutput='raw_values'),
                            [[9.09, 25], [0, 6.67]], decimal=2)
        assert_almost_equal(MAPE(y_true, y_pred, multioutput='raw_values'),
                            [[16.67, 40.00], [0, 14.29]], decimal=2)
        assert_almost_equal(RMSE(y_true, y_pred, multioutput='raw_values'),
                            [[0.5, 0.2], [0, 1]], decimal=2)
        assert_almost_equal(MSE(y_true, y_pred, multioutput='raw_values'),
                            [[0.25, 0.04], [0, 1]], decimal=2)

    def test_multioutput_array_metrics(self):
        y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
        y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]

        assert_array_almost_equal(MSE(y_true, y_pred, multioutput='raw_values'),
                                  [0.125, 0.5625], decimal=2)
        assert_array_almost_equal(MAE(y_true, y_pred, multioutput='raw_values'),
                                  [0.25, 0.625], decimal=2)
        assert_array_almost_equal(R2(y_true, y_pred, multioutput='raw_values'),
                                  [0.95, 0.93], decimal=2)

        assert_array_almost_equal(sMAPE(y_true, y_pred, multioutput='raw_values'),
                                  [4.09, 12.83], decimal=2)
        assert_array_almost_equal(MAPE(y_true, y_pred, multioutput='raw_values'),
                                  [7.78, 22.62], decimal=2)
        assert_array_almost_equal(MPE(y_true, y_pred, multioutput='raw_values'),
                                  [2.22, 5.95], decimal=2)
        assert_array_almost_equal(RMSE(y_true, y_pred, multioutput='raw_values'),
                                  [0.35, 0.75], decimal=2)

        assert_array_almost_equal(ME(y_true, y_pred, multioutput='raw_values'),
                                  [0., 0.12], decimal=2)
        assert_array_almost_equal(MSPE(y_true, y_pred, multioutput='raw_values'),
                                  [1.31, 9.16], decimal=2)
        assert_array_almost_equal(MDAPE(y_true, y_pred, multioutput='raw_values'),
                                  [5.56, 20.24], decimal=2)
        assert_array_almost_equal(sMDAPE(y_true, y_pred, multioutput='raw_values'),
                                  [2.63, 8.99], decimal=2)

    def test_large_array_metrics(self):
        y_true = np.ones((20000000, 4, 1), dtype=np.float32)
        y_pred = np.zeros((20000000, 4, 1), dtype=np.float32)

        assert_array_almost_equal(MAPE(y_true, y_pred, multioutput='raw_values'),
                                  [[100], [100], [100], [100]], decimal=2)
        assert_array_almost_equal(MSE(y_true, y_pred, multioutput='raw_values'),
                                  [[1], [1], [1], [1]], decimal=2)
        assert_array_almost_equal(MAE(y_true, y_pred, multioutput='raw_values'),
                                  [[1], [1], [1], [1]], decimal=2)

        assert_almost_equal(MSE(y_true, y_pred, multioutput='uniform_average'),
                            [1], decimal=2)
        assert_almost_equal(MAPE(y_true, y_pred, multioutput='uniform_average'),
                            [100], decimal=2)
        assert_almost_equal(MAE(y_true, y_pred, multioutput='uniform_average'),
                            [1], decimal=2)

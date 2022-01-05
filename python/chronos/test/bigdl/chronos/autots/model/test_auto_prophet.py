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

from bigdl.chronos.autots.model.auto_prophet import AutoProphet

import os
import numpy as np
import pandas as pd
import tempfile
from unittest import TestCase
from bigdl.orca.automl import hp


def get_data():
    seq_len = 480
    data = pd.DataFrame(pd.date_range('20130101', periods=seq_len), columns=['ds'])
    data.insert(1, 'y', np.random.rand(seq_len))
    expect_horizon = np.random.randint(40, 50)
    return data, expect_horizon


class TestAutoProphet(TestCase):
    def setUp(self) -> None:
        from bigdl.orca import init_orca_context
        init_orca_context(cores=4, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    def test_auto_prophet_fit(self):
        data, expect_horizon = get_data()
        auto_prophet = AutoProphet(metric="mse",
                                   changepoint_prior_scale=hp.loguniform(0.001, 0.5),
                                   seasonality_prior_scale=hp.loguniform(0.01, 10),
                                   holidays_prior_scale=hp.loguniform(0.01, 10),
                                   seasonality_mode=hp.choice(['additive', 'multiplicative']),
                                   changepoint_range=hp.uniform(0.8, 0.95),
                                   )

        auto_prophet.fit(data=data,
                         expect_horizon=expect_horizon,
                         n_sampling=1,
                         )
        best_model = auto_prophet.get_best_model()
        assert 0.001 <= best_model.changepoint_prior_scale <= 0.5
        assert 0.01 <= best_model.seasonality_prior_scale <= 10
        assert 0.01 <= best_model.holidays_prior_scale <= 10
        assert best_model.seasonality_mode in ['additive', 'multiplicative']
        assert 0.8 <= best_model.changepoint_range <= 0.95

    def test_auto_prophet_predict_evaluate(self):
        data, expect_horizon = get_data()
        auto_prophet = AutoProphet(metric="mse",
                                   changepoint_prior_scale=hp.loguniform(0.001, 0.5),
                                   seasonality_prior_scale=hp.loguniform(0.01, 10),
                                   holidays_prior_scale=hp.loguniform(0.01, 10),
                                   seasonality_mode=hp.choice(['additive', 'multiplicative']),
                                   changepoint_range=hp.uniform(0.8, 0.95)
                                   )

        auto_prophet.fit(data=data,
                         cross_validation=False,
                         expect_horizon=expect_horizon,
                         n_sampling=1,
                         )

        auto_prophet.predict(horizon=1, freq="D")
        test_data = pd.DataFrame(pd.date_range('20150101', periods=10),
                                 columns=['ds'])
        test_data.insert(1, 'y', np.random.rand(10))
        auto_prophet.evaluate(test_data)

    def test_auto_prophet_save_load(self):
        data, expect_horizon = get_data()
        auto_prophet = AutoProphet(metric="mse",
                                   changepoint_prior_scale=hp.loguniform(0.001, 0.5),
                                   seasonality_prior_scale=hp.loguniform(0.01, 10),
                                   holidays_prior_scale=hp.loguniform(0.01, 10),
                                   seasonality_mode=hp.choice(['additive', 'multiplicative']),
                                   changepoint_range=hp.uniform(0.8, 0.95)
                                   )

        auto_prophet.fit(data=data,
                         expect_horizon=expect_horizon,
                         n_sampling=1,
                         )
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "json")
            auto_prophet.save(ckpt_name)
            pred = auto_prophet.predict(horizon=10, freq="D")
            auto_prophet.restore(ckpt_name)
            pred_old = auto_prophet.predict(horizon=10, freq="D")
            new_auto_prophet = AutoProphet(load_dir=ckpt_name)
            pred_new = new_auto_prophet.predict(horizon=10, freq="D")
            np.testing.assert_almost_equal(pred.yhat.values, pred_new.yhat.values)
            np.testing.assert_almost_equal(pred.yhat.values, pred_old.yhat.values)

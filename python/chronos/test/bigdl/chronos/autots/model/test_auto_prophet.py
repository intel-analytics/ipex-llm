#
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

from zoo.chronos.autots.model.auto_prophet import AutoProphet

import numpy as np
import pandas as pd
from unittest import TestCase
from zoo.automl.recipe.base import Recipe
from zoo.orca.automl import hp


def get_data():
    seq_len = 480
    data = pd.DataFrame(pd.date_range('20130101', periods=seq_len), columns=['ds'])
    data.insert(1, 'y', np.random.rand(seq_len))
    horizon = np.random.randint(2, 50)
    validation_data = pd.DataFrame(pd.date_range('20140426', periods=horizon), columns=['ds'])
    validation_data.insert(1, 'y', np.random.rand(horizon))
    return data, validation_data


class TestAutoProphet(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        data, validation_data = get_data()
        auto_prophet = AutoProphet(metric="mse",
                                   changepoint_prior_scale=hp.loguniform(0.001, 0.5),
                                   seasonality_prior_scale=hp.loguniform(0.01, 10),
                                   holidays_prior_scale=hp.loguniform(0.01, 10),
                                   seasonality_mode=hp.choice(['additive', 'multiplicative']),
                                   changepoint_range=hp.uniform(0.8, 0.95)
                                   )

        auto_prophet.fit(data=data,
                         validation_data=validation_data,
                         epochs=1,
                         n_sampling=1,
                         )
        best_model = auto_prophet.get_best_model()
        assert 0.001 <= best_model.changepoint_prior_scale <= 0.5
        assert 0.01 <= best_model.seasonality_prior_scale <= 10
        assert 0.01 <= best_model.holidays_prior_scale <= 10
        assert best_model.seasonality_mode in ['additive', 'multiplicative']
        assert 0.8 <= best_model.changepoint_range <= 0.95

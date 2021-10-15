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
import pandas as pd
import tempfile
import os

from bigdl.chronos.forecaster.prophet_forecaster import ProphetForecaster
from unittest import TestCase
import pytest


def create_data():
    seq_len = 400
    data = pd.DataFrame(pd.date_range('20130101', periods=seq_len), columns=['ds'])
    data.insert(1, 'y', np.random.rand(seq_len))
    horizon = np.random.randint(2, 50)
    validation_data = pd.DataFrame(pd.date_range('20140426', periods=horizon), columns=['ds'])
    validation_data.insert(1, 'y', np.random.rand(horizon))
    return data, validation_data


class TestChronosModelProphetForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_prophet_forecaster_fit_eval_pred(self):
        data, validation_data = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )
        train_loss = forecaster.fit(data, validation_data)
        test_pred = forecaster.predict(validation_data.shape[0])
        assert test_pred.shape[0] == validation_data.shape[0]
        test_mse = forecaster.evaluate(validation_data)

    def test_prophet_forecaster_save_restore(self):
        data, validation_data = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )
        train_loss = forecaster.fit(data, validation_data)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "json")
            test_pred_save = forecaster.predict(validation_data.shape[0])
            forecaster.save(ckpt_name)
            forecaster.restore(ckpt_name)
            test_pred_restore = forecaster.predict(validation_data.shape[0])
        assert (test_pred_save['yhat'] == test_pred_restore['yhat']).all()

    def test_prophet_forecaster_runtime_error(self):
        data, validation_data = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling predict!"):
            forecaster.predict(horizon=validation_data.shape[0])

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling save!"):
            model_file = "tmp.json"
            forecaster.save(model_file)

    def test_prophet_forecaster_shape_error(self):
        data, validation_data = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )

        with pytest.raises(AssertionError):
            forecaster.fit(data[['ds']], validation_data)

        with pytest.raises(AssertionError):
            forecaster.fit(data, validation_data[['ds']])

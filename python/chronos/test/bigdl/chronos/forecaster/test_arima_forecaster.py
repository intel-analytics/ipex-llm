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
import tempfile
import os

from bigdl.chronos.forecaster.arima_forecaster import ARIMAForecaster
from unittest import TestCase
import pytest


def create_data():
    seq_len = 400
    data = np.random.rand(seq_len)
    horizon = np.random.randint(2, 50)
    validation_data = np.random.rand(horizon)
    return data, validation_data


class TestChronosModelARIMAForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_arima_forecaster_fit_eval_pred(self):
        data, validation_data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )
        train_loss = forecaster.fit(data, validation_data)
        test_pred = forecaster.predict(len(validation_data))
        assert len(test_pred) == len(validation_data)
        # test rolling predict
        test_rolling_pred = forecaster.predict(len(validation_data), rolling=True)
        assert len(test_rolling_pred) == len(validation_data)
        test_mse = forecaster.evaluate(validation_data)

    def test_arima_forecaster_save_restore(self):
        data, validation_data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )
        train_loss = forecaster.fit(data, validation_data)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "pkl")
            test_pred_save = forecaster.predict(len(validation_data))
            forecaster.save(ckpt_name)
            forecaster.restore(ckpt_name)
            test_pred_restore = forecaster.predict(len(validation_data))
        np.testing.assert_almost_equal(test_pred_save, test_pred_restore)

    def test_arima_forecaster_runtime_error(self):
        data, validation_data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling predict!"):
            forecaster.predict(horizon=len(validation_data))

        with pytest.raises(Exception,
                           match="Input invalid validation_data of None"):
            forecaster.evaluate(validation_data=None)

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling evaluate!"):
            forecaster.evaluate(validation_data=validation_data)

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling save!"):
            model_file = "tmp.pkl"
            forecaster.save(model_file)

    def test_arima_forecaster_shape_error(self):
        data, validation_data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )

        with pytest.raises(AssertionError):
            forecaster.fit(data.reshape(-1, 1), validation_data)

        with pytest.raises(AssertionError):
            forecaster.fit(data, validation_data.reshape(-1, 1))

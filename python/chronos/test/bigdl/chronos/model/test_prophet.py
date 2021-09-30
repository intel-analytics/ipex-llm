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

import pytest

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.model.prophet import ProphetModel
import numpy as np
import os
from numpy.testing import assert_array_almost_equal
import pandas as pd


class TestProphetModel(ZooTestCase):

    def setup_method(self, method):
        self.seq_len = 480
        self.config = {
            "changepoint_prior_scale": np.exp(np.random.uniform(np.log(0.001), np.log(0.5))),
            "seasonality_prior_scale": np.exp(np.random.uniform(np.log(0.01), np.log(10))),
            "holidays_prior_scale": np.exp(np.random.uniform(np.log(0.01), np.log(10))),
            "seasonality_mode": np.random.choice(['additive', 'multiplicative']),
            "changepoint_range": np.random.uniform(0.8, 0.95)
        }
        self.model = ProphetModel()
        self.data = pd.DataFrame(pd.date_range('20130101', periods=self.seq_len), columns=['ds'])
        self.data.insert(1, 'y', np.random.rand(self.seq_len))
        self.horizon = np.random.randint(2, 50)
        self.validation_data = pd.DataFrame(pd.date_range('20140426', periods=self.horizon),
                                            columns=['ds']
                                            )
        self.validation_data.insert(1, 'y', np.random.rand(self.horizon))

    def teardown_method(self, method):
        del self.model
        del self.data
        del self.validation_data

    def test_prophet(self):
        # test fit_eval
        evaluate_result = self.model.fit_eval(data=self.data,
                                              validation_data=self.validation_data,
                                              **self.config)
        # test predict
        result = self.model.predict(horizon=self.horizon)
        assert result.shape[0] == self.horizon
        # test evaluate
        evaluate_result = self.model.evaluate(target=self.validation_data,
                                              metrics=['mae', 'smape'])
        assert len(evaluate_result) == 2

    def test_error(self):

        with pytest.raises(ValueError, match="We don't support input data currently"):
            self.model.evaluate(target=self.validation_data, data=1)

        with pytest.raises(ValueError, match="Input invalid target of None"):
            self.model.evaluate(target=None)

        with pytest.raises(Exception,
                           match="Needs to call fit_eval or restore first before calling predict"):
            self.model.predict()

        with pytest.raises(Exception,
                           match="Needs to call fit_eval or restore first before calling evaluate"):
            self.model.evaluate(target=self.validation_data)

        with pytest.raises(Exception,
                           match="Needs to call fit_eval or restore first before calling save"):
            model_file = "tmp.json"
            self.model.save(model_file)

    def test_save_restore(self):
        self.model.fit_eval(data=self.data,
                            validation_data=self.validation_data,
                            **self.config)
        result_save = self.model.predict(horizon=self.horizon)
        model_file = "tmp.json"

        self.model.save(model_file)
        assert os.path.isfile(model_file)
        new_model = ProphetModel()
        new_model.restore(model_file)
        assert new_model.model
        result_restore = new_model.predict(horizon=self.horizon)
        assert_array_almost_equal(result_save['yhat'], result_restore['yhat'], decimal=2), \
            "Prediction values are not the same after restore: " \
            "predict before is {}, and predict after is {}".format(result_save, result_restore)
        os.remove(model_file)


if __name__ == '__main__':
    pytest.main([__file__])

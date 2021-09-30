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
from bigdl.chronos.model.tcmf_model import TCMF
import numpy as np
import os
from numpy.testing import assert_array_almost_equal
import pandas as pd


class TestTCMF(ZooTestCase):

    def setup_method(self, method):
        self.seq_len = 480
        self.num_samples = 300
        self.config = {
            "y_iters": 1,
            "init_FX_epoch": 1,
            "max_FX_epoch": 1,
            "max_TCN_epoch": 1,
            "alt_iters": 2,
        }
        self.model = TCMF()
        self.Ymat = np.random.rand(self.num_samples, self.seq_len)
        self.horizon = np.random.randint(2, 50)

    def teardown_method(self, method):
        del self.model
        del self.Ymat

    def test_tcmf(self):
        self.model.fit_eval(data=(self.Ymat, None), **self.config)
        # test predict
        result = self.model.predict(horizon=self.horizon)
        assert result.shape[1] == self.horizon
        # test evaluate
        target = np.random.rand(self.num_samples, self.horizon)
        evaluate_result = self.model.evaluate(y=target, metrics=['mae', 'smape'])
        assert len(evaluate_result) == 2
        assert len(evaluate_result[0]) == self.horizon
        assert len(evaluate_result[1]) == self.horizon
        # test fit_incremental
        Ymat_before = self.model.model.Ymat
        self.model.fit_incremental(target)
        Ymat_after = self.model.model.Ymat
        assert Ymat_after.shape[1] - Ymat_before.shape[1] == self.horizon
        incr_result = self.model.predict(horizon=self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, result, incr_result)

    def test_tcmf_covariates_dti(self):
        # test wrong format in fit
        with pytest.raises(ValueError, match="Input covariates must be a ndarray. Got"):
            self.model.fit_eval(data=(self.Ymat, None),
                                covariates='None',
                                **self.config)

        with pytest.raises(ValueError, match="The second dimension shape of covariates should be"):
            self.model.fit_eval(data=(self.Ymat, None),
                                covariates=np.random.randn(3, self.seq_len - 1),
                                **self.config)

        with pytest.raises(ValueError, match="You should input a 2-D ndarray of covariates."):
            self.model.fit_eval(data=(self.Ymat, None),
                                covariates=np.random.randn(3, 4, 5),
                                **self.config)

        with pytest.raises(ValueError, match="Input dti must be a pandas DatetimeIndex. Got"):
            self.model.fit_eval(data=(self.Ymat, None),
                                covariates=np.random.randn(3, self.seq_len),
                                dti='None',
                                **self.config)

        with pytest.raises(ValueError, match="Input dti length should be equal to"):
            self.model.fit_eval(data=(self.Ymat, None),
                                covariates=np.random.randn(3, self.seq_len),
                                dti=pd.date_range('20130101', periods=self.seq_len-1),
                                **self.config)

        # valid covariates and dti in fit_eval
        self.model.fit_eval(data=(self.Ymat, None),
                            covariates=np.random.rand(3, self.seq_len),
                            dti=pd.date_range('20130101', periods=self.seq_len),
                            **self.config)

        # inconsistent covariates and dti
        with pytest.raises(ValueError, match="Find valid covariates in fit but invalid covariates "
                                             "in predict."):
            self.model.predict(horizon=self.horizon)

        with pytest.raises(ValueError, match="be the same as the input covariates number in fit."):
            self.model.predict(horizon=self.horizon,
                               future_covariates=np.random.randn(2, self.horizon),
                               )
        with pytest.raises(ValueError, match="Find valid dti in fit but invalid dti in"):
            self.model.predict(horizon=self.horizon,
                               future_covariates=np.random.randn(3, self.horizon),
                               )

        with pytest.raises(ValueError, match="Find valid covariates in fit but invalid covariates "
                                             "in fit_incremental."):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon))

        with pytest.raises(ValueError, match="be the same as the input covariates number in fit."):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon),
                                       covariates_new=np.random.randn(2, self.horizon),)
        with pytest.raises(ValueError, match="Find valid dti in fit but invalid dti in"):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon),
                                       covariates_new=np.random.randn(3, self.horizon), )

        self.model.predict(horizon=self.horizon,
                           future_covariates=np.random.randn(3, self.horizon),
                           future_dti=pd.date_range('20130101', periods=self.horizon),
                           )
        self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon),
                                   covariates_new=np.random.randn(3, self.horizon),
                                   dti_new=pd.date_range('20130101', periods=self.horizon),)

    def test_error(self):
        with pytest.raises(ValueError, match="We don't support input x directly"):
            self.model.predict(x=1)

        with pytest.raises(ValueError, match="We don't support input x directly"):
            self.model.evaluate(x=1, y=np.random.rand(self.num_samples, self.horizon))

        with pytest.raises(ValueError, match="Input invalid y of None"):
            self.model.evaluate(y=None)

        with pytest.raises(Exception,
                           match="Needs to call fit_eval or restore first before calling predict"):
            self.model.predict(x=None)

        with pytest.raises(Exception,
                           match="Needs to call fit_eval or restore first before calling predict"):
            self.model.evaluate(y=np.random.rand(self.num_samples, self.horizon))

        with pytest.raises(ValueError, match="Input invalid x of None"):
            self.model.fit_incremental(x=None)

        with pytest.raises(Exception, match="Needs to call fit_eval or restore first before "
                                            "calling fit_incremental"):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon))

        self.model.fit_eval(data=(self.Ymat, None), **self.config)
        with pytest.raises(Exception, match=f"Expected incremental input with {self.num_samples} "
                                            f"time series, got {self.num_samples - 1} instead"):
            self.model.fit_incremental(x=np.random.rand(self.num_samples - 1, self.horizon))

        with pytest.raises(ValueError, match="but invalid covariates in fit. "):
            self.model.predict(horizon=self.horizon,
                               future_covariates=np.random.randn(3, self.horizon),
                               )
        with pytest.raises(ValueError, match="but invalid dti in fit. "):
            self.model.predict(horizon=self.horizon,
                               future_dti=pd.date_range('20130101', periods=self.horizon),
                               )
        with pytest.raises(ValueError, match="but invalid covariates in fit. "):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon),
                                       covariates_new=np.random.randn(3, self.horizon),)
        with pytest.raises(ValueError, match="but invalid dti in fit. "):
            self.model.fit_incremental(x=np.random.rand(self.num_samples, self.horizon),
                                       dti_new=pd.date_range('20130101', periods=self.horizon), )

    def test_save_restore(self):
        self.model.fit_eval(data=(self.Ymat, None), **self.config)
        result_save = self.model.predict(horizon=self.horizon)
        model_file = "tmp.pkl"
        self.model.save(model_file)
        assert os.path.isfile(model_file)
        new_model = TCMF()
        new_model.restore(model_file)
        assert new_model.model
        result_restore = new_model.predict(horizon=self.horizon)
        assert_array_almost_equal(result_save, result_restore, decimal=2), \
            "Prediction values are not the same after restore: " \
            "predict before is {}, and predict after is {}".format(result_save, result_restore)
        os.remove(model_file)


if __name__ == '__main__':
    pytest.main([__file__])

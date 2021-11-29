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
import os
import shutil
import tempfile

import pytest

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.model.Seq2Seq import LSTMSeq2Seq
from bigdl.chronos.autots.deprecated.feature.time_sequence import TimeSequenceFeatureTransformer
from numpy.testing import assert_array_almost_equal
import pandas as pd
import numpy as np


class TestSeq2Seq(ZooTestCase):

    def setup_method(self, method):
        # super().setup_method(method)
        self.train_data = pd.DataFrame(data=np.random.randn(64, 4))
        self.val_data = pd.DataFrame(data=np.random.randn(16, 4))
        self.test_data = pd.DataFrame(data=np.random.randn(16, 4))

        self.past_seq_len = 6
        self.future_seq_len_1 = 1
        self.future_seq_len_2 = 2

        # use roll method in time_sequence
        self.feat = TimeSequenceFeatureTransformer()

        self.config = {
            'batch_size': 32,
            'epochs': 1,
            'latent_dim': 8
        }

        self.model_1 = LSTMSeq2Seq(check_optional_config=False,
                                   future_seq_len=self.future_seq_len_1)
        self.model_2 = LSTMSeq2Seq(check_optional_config=False,
                                   future_seq_len=self.future_seq_len_2)

        self.fitted = False
        self.predict_1 = None
        self.predict_2 = None

    def teardown_method(self, method):
        pass

    def test_fit_eval_1(self):
        x_train_1, y_train_1 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_1)
        print("fit_eval_future_seq_len_1:",
              self.model_1.fit_eval((x_train_1, y_train_1), **self.config))
        assert self.model_1.past_seq_len == 6
        assert self.model_1.feature_num == 4
        assert self.model_1.future_seq_len == 1
        assert self.model_1.target_col_num == 1

    def test_fit_eval(self):
        past_seq_len = 6
        future_seq_len = 2
        input_dim = 5
        output_dim = 4
        x_train = np.random.rand(100, past_seq_len, input_dim)
        y_train = np.random.rand(100, future_seq_len, output_dim)
        x_test = np.random.rand(100, past_seq_len, input_dim)
        y_test = np.random.rand(100, future_seq_len, output_dim)
        model = LSTMSeq2Seq(check_optional_config=False,
                            future_seq_len=future_seq_len)
        model_config = {
            'batch_size': 32,
            'epochs': 1,
            'latent_dim': 8,
            'dropout': 0.2
        }
        model.fit_eval((x_train, y_train), **model_config)
        y_pred = model.predict(x_test)
        rmse, smape = model.evaluate(x=x_test, y=y_test, metric=["rmse", "smape"])
        assert rmse.shape == smape.shape
        assert rmse.shape == (future_seq_len, output_dim)

        assert model.past_seq_len == past_seq_len
        assert model.future_seq_len == future_seq_len
        assert model.feature_num == input_dim
        assert model.target_col_num == output_dim
        assert y_pred.shape == y_test.shape

    def test_fit_eval_2(self):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        print("fit_eval_future_seq_len_2:",
              self.model_2.fit_eval((x_train_2, y_train_2), **self.config))
        assert self.model_2.future_seq_len == 2

        self.fitted = True

    def test_evaluate_1(self):
        x_train_1, y_train_1 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_1)
        x_val_1, y_val_1 = self.feat._roll_train(self.val_data,
                                                 past_seq_len=self.past_seq_len,
                                                 future_seq_len=self.future_seq_len_1)

        self.model_1.fit_eval((x_train_1, y_train_1), **self.config)

        print("evaluate_future_seq_len_1:", self.model_1.evaluate(x_val_1,
                                                                  y_val_1,
                                                                  metric=['mse',
                                                                          'r2']))

    def test_evaluate_2(self):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        x_val_2, y_val_2 = self.feat._roll_train(self.val_data,
                                                 past_seq_len=self.past_seq_len,
                                                 future_seq_len=self.future_seq_len_2)

        self.model_2.fit_eval((x_train_2, y_train_2), **self.config)

        print("evaluate_future_seq_len_2:", self.model_2.evaluate(x_val_2,
                                                                  y_val_2,
                                                                  metric=['mse',
                                                                          'r2']))

    def test_predict_1(self):
        x_train_1, y_train_1 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_1)
        x_test_1 = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        self.model_1.fit_eval((x_train_1, y_train_1), **self.config)

        predict_1 = self.model_1.predict(x_test_1)
        assert predict_1.shape == (x_test_1.shape[0], self.future_seq_len_1)

    def test_predict_2(self):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        x_test_2 = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        self.model_2.fit_eval((x_train_2, y_train_2), **self.config)

        predict_2 = self.model_2.predict(x_test_2)
        assert predict_2.shape == (x_test_2.shape[0], self.future_seq_len_2)

    def test_save_restore_single_step(self):
        future_seq_len = 1
        x_train, y_train = self.feat._roll_train(self.train_data,
                                                 past_seq_len=self.past_seq_len,
                                                 future_seq_len=future_seq_len)
        x_test = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        model = LSTMSeq2Seq(future_seq_len=future_seq_len)
        model.fit_eval((x_train, y_train), **self.config)

        predict_before = model.predict(x_test)
        new_model = LSTMSeq2Seq()

        ckpt = os.path.join("/tmp", "seq2seq.ckpt")
        model.save(ckpt)
        new_model.restore(ckpt)
        predict_after = new_model.predict(x_test)
        assert_array_almost_equal(predict_before, predict_after, decimal=2), \
            "Prediction values are not the same after restore: " \
            "predict before is {}, and predict after is {}".format(predict_before, predict_after)
        new_config = {'epochs': 1, 'latent_dim': 8}
        new_model.fit_eval((x_train, y_train), **new_config)
        os.remove(ckpt)

    def test_save_restore_multistep(self):
        future_seq_len = np.random.randint(2, 6)
        x_train, y_train = self.feat._roll_train(self.train_data,
                                                 past_seq_len=self.past_seq_len,
                                                 future_seq_len=future_seq_len)
        x_test = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        model = LSTMSeq2Seq(future_seq_len=future_seq_len)
        model.fit_eval((x_train, y_train), **self.config)

        predict_before = model.predict(x_test)
        new_model = LSTMSeq2Seq()

        ckpt = os.path.join("/tmp", "seq2seq.ckpt")
        model.save(ckpt)
        new_model.restore(ckpt)
        predict_after = new_model.predict(x_test)
        assert_array_almost_equal(predict_before, predict_after, decimal=2), \
            "Prediction values are not the same after restore: " \
            "predict before is {}, and predict after is {}".format(predict_before, predict_after)
        new_config = {'epochs': 1, 'latent_dim': 8}
        new_model.fit_eval((x_train, y_train), **new_config)
        os.remove(ckpt)

    def test_predict_with_uncertainty(self,):
        future_seq_len = np.random.randint(2, 6)
        x_train, y_train = self.feat._roll_train(self.train_data,
                                                 past_seq_len=self.past_seq_len,
                                                 future_seq_len=future_seq_len)
        x_test = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        model = LSTMSeq2Seq(future_seq_len=future_seq_len)
        model.fit_eval((x_train, y_train), mc=True, **self.config)

        prediction, uncertainty = model.predict_with_uncertainty(x_test, n_iter=2)
        assert prediction.shape == (x_test.shape[0], future_seq_len)
        assert uncertainty.shape == (x_test.shape[0], future_seq_len)
        assert np.any(uncertainty)

        new_model = LSTMSeq2Seq()

        ckpt = os.path.join("/tmp", "seq2seq.ckpt")
        model.save(ckpt)
        new_model.restore(ckpt)
        prediction_after, uncertainty_after = new_model.predict_with_uncertainty(x_test, n_iter=2)
        assert prediction_after.shape == (x_test.shape[0], future_seq_len)
        assert uncertainty_after.shape == (x_test.shape[0], future_seq_len)
        assert np.any(uncertainty_after)

        os.remove(ckpt)


if __name__ == '__main__':
    pytest.main([__file__])

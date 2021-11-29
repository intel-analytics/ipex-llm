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
import numpy as np
from bigdl.chronos.autots.deprecated.feature.time_sequence import TimeSequenceFeatureTransformer
import tensorflow as tf
import pandas as pd

from bigdl.chronos.forecaster.mtnet_forecaster import MTNetForecaster
from unittest import TestCase


class TestChronosModelMTNetForecaster(TestCase):

    def setUp(self):
        tf.keras.backend.clear_session()
        self.ft = TimeSequenceFeatureTransformer()
        self.create_data()

    def tearDown(self):
        pass

    def create_data(self):
        def gen_train_sample(data, past_seq_len, future_seq_len):
            data = pd.DataFrame(data)
            x, y = self.ft._roll_train(data,
                                       past_seq_len=past_seq_len,
                                       future_seq_len=future_seq_len
                                       )
            return x, y

        def gen_test_sample(data, past_seq_len):
            test_data = pd.DataFrame(data)
            x = self.ft._roll_test(test_data, past_seq_len=past_seq_len)
            return x

        self.long_num = 4
        self.time_step = 1
        look_back = (self.long_num + 1) * self.time_step
        look_forward = 1
        self.x_train, self.y_train = gen_train_sample(data=np.random.randn(
            64, 4), past_seq_len=look_back, future_seq_len=look_forward)
        self.x_val, self.y_val = gen_train_sample(data=np.random.randn(16, 4),
                                                  past_seq_len=look_back,
                                                  future_seq_len=look_forward)
        self.x_test = gen_test_sample(data=np.random.randn(16, 4),
                                      past_seq_len=look_back)

    def test_forecast_mtnet(self):
        # TODO hacking to fix a bug
        target_dim = 1
        model = MTNetForecaster(target_dim=target_dim,
                                feature_dim=self.x_train.shape[-1],
                                long_series_num=self.long_num,
                                series_length=self.time_step
                                )
        x_train_long, x_train_short = model.preprocess_input(self.x_train)
        x_val_long, x_val_short = model.preprocess_input(self.x_val)
        x_test_long, x_test_short = model.preprocess_input(self.x_test)

        model.fit([x_train_long, x_train_short],
                  self.y_train,
                  validation_data=([x_val_long, x_val_short], self.y_val),
                  batch_size=32,
                  distributed=False)
        assert model.evaluate([x_val_long, x_val_short], self.y_val)
        predict_result = model.predict([x_test_long, x_test_short])
        assert predict_result.shape == (self.x_test.shape[0], target_dim)


if __name__ == "__main__":
    pytest.main([__file__])

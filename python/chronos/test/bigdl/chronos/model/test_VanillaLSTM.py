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
from unittest import TestCase
from bigdl.chronos.model.VanillaLSTM import VanillaLSTM
import numpy as np
import tempfile
import os


def create_data():
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 7
    input_feature_dim = 4
    output_dim = np.random.randint(1, 5)

    def get_x_y(num_samples):
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim)
        y = np.random.randn(num_samples, output_dim)
        return x, y

    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return train_data, val_data, test_data


class TestVanillaLSTM(TestCase):
    train_data, val_data, test_data = create_data()
    model = VanillaLSTM()

    def test_fit_evaluate(self):
        config = {"batch_size": 128}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)
        mse, smape = self.model.evaluate(self.val_data[0], self.val_data[1],
                                         metrics=["mse", "smape"])

    def test_config(self):
        config = {"lstm_units": [128] * 2,
                  "dropouts": [0.2] * 2}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)

        config = {"lstm_units": 128,
                  "dropouts": 0.2}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)

        with pytest.raises(ValueError):
            config = {"lstm_units": 0.1,
                      "dropouts": 0.2}
            self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)

        with pytest.raises(ValueError):
            config = {"lstm_units": [128] * 2,
                      "dropouts": [0.2] * 3}
            self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)

        with pytest.raises(ValueError):
            config = {"lstm_units": 128,
                      "dropouts": [0.2] * 2}
            self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)

    def test_predict_save_restore(self):
        model = VanillaLSTM()
        config = {"lstm_units": [128] * 2,
                  "dropouts": [0.2] * 2,
                  "batch_size": 128}
        model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)
        pred = model.predict(self.test_data[0])
        assert pred.shape == self.test_data[1].shape
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            model.save(ckpt_name)
            model_1 = VanillaLSTM()
            model_1.restore(ckpt_name)
            pred_1 = model_1.predict(self.test_data[0])
            assert np.allclose(pred, pred_1)

    def test_predict_with_uncertainty(self):
        config = {"batch_size": 128}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, **config)
        prediction, uncertainty = self.model.predict_with_uncertainty(self.test_data[0], n_iter=100)
        assert prediction.shape == self.test_data[1].shape
        assert uncertainty.shape == self.test_data[1].shape
        assert np.any(uncertainty)


if __name__ == '__main__':
    pytest.main([__file__])

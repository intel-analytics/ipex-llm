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

from unittest import TestCase
from bigdl.chronos.model.tcn import TCNPytorch
import numpy as np
import tempfile
import os


def create_data():
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 24
    input_feature_dim = 4
    output_time_steps = np.random.randint(1, 5)
    output_feature_dim = np.random.randint(1, 4)

    def get_x_y(num_samples):
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim)
        y = np.random.randn(num_samples, output_time_steps, output_feature_dim)
        return x, y

    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return train_data, val_data, test_data


class TestTcn(TestCase):
    train_data, val_data, test_data = create_data()
    model = TCNPytorch()

    def test_fit_evaluate(self):
        config = {"batch_size": 128}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data,
                            metric="mse",
                            **config)
        mse, smape = self.model.evaluate(self.val_data[0],
                                         self.val_data[1],
                                         metrics=["mse", "smape"])
        assert len(mse) == self.val_data[1].shape[-2]
        assert len(mse[0]) == self.val_data[1].shape[-1]
        assert len(smape) == self.val_data[1].shape[-2]
        assert len(smape[0]) == self.val_data[1].shape[-1]

    def test_predict_save_restore(self):
        config = {"batch_size": 128}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data,
                            metric="mse",
                            **config)
        pred = self.model.predict(self.test_data[0])
        assert pred.shape == self.test_data[1].shape
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            self.model.save(ckpt_name)
            model_1 = TCNPytorch()
            model_1.restore(ckpt_name)
            pred_1 = model_1.predict(self.test_data[0])
            assert np.allclose(pred, pred_1)

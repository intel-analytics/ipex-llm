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
from pathlib import Path
import shutil

import pytest
from unittest import TestCase
from bigdl.chronos.data import TSDataset
import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal
from ... import op_tf2
from bigdl.chronos.utils import LazyImport
tf = LazyImport('tensorflow')
MTNetKeras = LazyImport('bigdl.chronos.model.tf2.MTNet_keras.MTNetKeras')


def create_data():
    lookback = 3
    horizon = 1
    def get_data(num_samples):
        values = np.random.randn(num_samples)
        df = pd.DataFrame({'timestep': pd.date_range(start='2010-01-01',
                                                     freq='m',
                                                     periods=num_samples),
                           'value 1': values,
                           'value 2': values,
                           'value 3': values,
                           'value 4': values})
        return df
    tsdata_train = TSDataset.from_pandas(get_data(32),
                                         target_col=['value 1', 'value 2', 'value 3', 'value 4'],
                                         dt_col='timestep',
                                         with_split=False) 
    tsdata_test = TSDataset.from_pandas(get_data(16),
                                        target_col=['value 1', 'value 2', 'value 3', 'value 4'],
                                        dt_col='timestep',
                                        with_split=False)
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.roll(lookback=lookback, horizon=horizon)
    return tsdata_train, tsdata_test


@op_tf2
class TestMTNetKeras(TestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        train_data, test_data = create_data()
        self.x_train, y_train = train_data.to_numpy()
        self.y_train = y_train.reshape(y_train.shape[0], y_train.shape[-1])
        self.x_val, y_val = test_data.to_numpy()
        self.y_val = y_val.reshape(y_val.shape[0], y_val.shape[-1])
        self.x_test, _ = test_data.to_numpy()
        self.model = MTNetKeras()
        self.config = {"long_num": 2,
                       "time_step": 1,
                       "ar_window": 1, # np.random.randint(1, 3),
                       "cnn_height": 1, # np.random.randint(1, 3),
                       "cnn_hid_size": 2,
                       "rnn_hid_sizes": [2, 2],
                       "batch_size": 32,
                       "epochs": 1}

    def teardown_method(self, method):
        pass

    def test_fit_evaluate(self):
        self.model.fit_eval(data=(self.x_train, self.y_train),
                            validation_data=(self.x_val, self.y_val),
                            **self.config)
        self.model.evaluate(self.x_val, self.y_val, batch_size=32)

    def test_save_restore(self):
        import os
        self.model.fit_eval(data=(self.x_train, self.y_train),
                            validation_data=(self.x_val, self.y_val),
                            **self.config)
        y_pred = self.model.predict(self.x_test)
        assert y_pred.shape == (self.x_test.shape[0], self.y_val.shape[-1])
        dirname = Path("savedroot")
        dirname.mkdir(exist_ok=True)
        restored_model = MTNetKeras()
        ckpt = os.path.join(dirname, "mtnet.ckpt")
        self.model.save(checkpoint_file=ckpt)
        restored_model.restore(checkpoint_file=ckpt)
        predict_after = restored_model.predict(self.x_test)
        assert_array_almost_equal(y_pred, predict_after, decimal=2), \
            "Prediction values are not the same after restore: " \
            "predict before is {}, and predict after is {}".format(y_pred, predict_after)
        restored_model.fit_eval((self.x_train, self.y_train), epochs=1)
        restored_model.evaluate(self.x_val, self.y_val)
        os.remove(ckpt)

    def test_predict_with_uncertainty(self):
        self.model.fit_eval(data=(self.x_train, self.y_train),
                            validation_data=(self.x_val, self.y_val),
                            mc=True,
                            **self.config)
        pred, uncertainty = self.model.predict_with_uncertainty(self.x_test, n_iter=2)
        assert pred.shape == (self.x_test.shape[0], self.y_val.shape[-1])
        assert uncertainty.shape == pred.shape
        # assert np.any(uncertainty) It may happen that all results are dropped out.


if __name__ == '__main__':
    pytest.main([__file__])

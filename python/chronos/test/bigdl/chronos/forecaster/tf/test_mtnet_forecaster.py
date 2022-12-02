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
import pandas as pd

from bigdl.chronos.data import TSDataset
from bigdl.chronos.utils import LazyImport
tf = LazyImport('tensorflow')
from unittest import TestCase
from test.bigdl.chronos import op_tf2


def create_data():
    lookback = 5
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
    tsdata_train = TSDataset.from_pandas(get_data(64),
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
class TestChronosModelMTNetForecaster(TestCase):

    def setUp(self):
        tf.keras.backend.clear_session()

    def tearDown(self):
        pass

    def test_forecast_mtnet(self):
        # TODO hacking to fix a bug
        train_data, test_data = create_data()
        self.x_train, y_train = train_data.to_numpy()
        self.y_train = y_train.reshape(y_train.shape[0], y_train.shape[-1])
        self.x_val, y_val = test_data.to_numpy()
        self.y_val = y_val.reshape(y_val.shape[0], y_val.shape[-1])
        self.x_test, y_test = test_data.to_numpy()
        self.y_test = y_test.reshape(y_test.shape[0], y_test.shape[-1])
        from bigdl.chronos.forecaster.tf.mtnet_forecaster import MTNetForecaster
        model = MTNetForecaster(target_dim=1,
                                feature_dim=self.x_train.shape[-1],
                                long_series_num=4,
                                series_length=1
                                )

        model.fit(data=(self.x_train, y_train),
                  validation_data=(self.x_val, y_val),
                  epochs=2,
                  batch_size=32)
        assert model.evaluate(data=(self.x_test, self.y_test))
        predict_result = model.predict(self.x_test)
        assert predict_result.shape == (self.x_test.shape[0], self.x_test.shape[-1])


if __name__ == "__main__":
    pytest.main([__file__])

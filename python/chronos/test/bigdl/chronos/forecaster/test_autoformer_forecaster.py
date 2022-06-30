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

from bigdl.chronos.forecaster.autoformer_forecaster import AutoformerForecaster
from bigdl.chronos.data import TSDataset
from unittest import TestCase
import pytest

def get_ts_df():
    sample_num = np.random.randint(1000, 1500)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num, freq="1s"),
                             "value": np.random.randn(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


def create_data(loader=False):
    df = get_ts_df()
    target = ["value", "extra feature"]
    tsdata_train, tsdata_val, tsdata_test =\
        TSDataset.from_pandas(df, dt_col="datetime", target_col=target,
                              with_split=True, test_ratio=0.1, val_ratio=0.1)
    if loader:
        train_loader = tsdata_train.to_torch_data_loader(roll=True, lookback=24, horizon=5,
                                                        time_enc=True, label_len=12)
        val_loader = tsdata_val.to_torch_data_loader(roll=True, lookback=24, horizon=5,
                                                    time_enc=True, label_len=12, shuffle=False)
        test_loader = tsdata_test.to_torch_data_loader(roll=True, lookback=24, horizon=5,
                                                    time_enc=True, label_len=12, shuffle=False,
                                                    is_predict=True)
        return train_loader, val_loader, test_loader
    else:
        train_data = tsdata_train.roll(lookback=24, horizon=5, time_enc=True, label_len=12).to_numpy()
        val_data = tsdata_val.roll(lookback=24, horizon=5, time_enc=True, label_len=12).to_numpy()
        test_data = tsdata_test.roll(lookback=24, horizon=5, time_enc=True, label_len=12).to_numpy()
        return train_data, val_data, test_data

class TestChronosModelTCNForecaster(TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_autoformer_forecaster_loader_fit_eval_pred(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(train_loader, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_loader)
        pred = forecaster.predict(test_loader)
        
    def test_autoformer_forecaster_array_fit_eval_pred(self):
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(train_data, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_data)
        pred = forecaster.predict(test_data)
    
    def test_autoformer_forecaster_tune(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          loss="mae",
                                          metrics=['mae', 'mse', 'mape'],
                                          lr=space.Real(0.001, 0.01, log=True))
        forecaster.tune(train_data, validation_data=val_data,
                        n_trials=2, target_metric='mse', direction="minimize")
    

    def test_autoformer_forecaster_save_load(self):
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(self.train_loader, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(self.val_loader)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "af.ckpt")
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            evaluate2 = forecaster.evaluate(self.val_loader)
        assert evaluate[0]['val_loss'] == evaluate2[0]['val_loss']


if __name__ == "__main__":
    train_data, val_data, test_data = create_data(loader=False)
    print(len(train_data))
    for x in train_data:
        print(x.shape)
    ut = TestChronosModelTCNForecaster()
    # ut.test_autoformer_forecaster_loader_fit_eval_pred()
    # ut.test_autoformer_forecaster_array_fit_eval_pred()
    ut.test_autoformer_forecaster_tune()
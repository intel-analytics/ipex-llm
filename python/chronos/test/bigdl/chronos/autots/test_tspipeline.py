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
from unittest import TestCase
import pytest
from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
TensorDataset = LazyImport('torch.utils.data.TensorDataset')
DataLoader = LazyImport('torch.utils.data.DataLoader')
TSPipeline = LazyImport('bigdl.chronos.autots.tspipeline.TSPipeline')
import os
import pandas as pd
import numpy as np

from bigdl.chronos.data import TSDataset

from .. import op_torch, op_inference

def train_data_creator(config):
    return DataLoader(TensorDataset(torch.randn(1000,
                                                config.get('past_seq_len', 10),
                                                config.get('input_feature_num', 2)),
                                    torch.randn(1000,
                                                config.get('future_seq_len', 2),
                                                config.get('output_feature_num', 2))),
                      batch_size=config.get('batch_size', 32), shuffle=True)

def valid_data_creator(config):
    return DataLoader(TensorDataset(torch.randn(1000,
                                                config.get('past_seq_len', 10),
                                                config.get('input_feature_num', 2)),
                                    torch.randn(1000,
                                                config.get('future_seq_len', 2),
                                                config.get('output_feature_num', 2))),
                      batch_size=config.get('batch_size', 32), shuffle=False)

def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value 1": np.random.randn(sample_num),
                             "value 2": np.random.randn(sample_num),
                             "id": np.array(['00'] * sample_num),
                             "extra feature 1": np.random.randn(sample_num),
                             "extra feature 2": np.random.randn(sample_num)})
    return train_df

def get_test_tsdataset():
    df = get_ts_df()
    return TSDataset.from_pandas(df,
                                 dt_col="datetime",
                                 target_col=["value 1", "value 2"],
                                 extra_feature_col=["extra feature 1", "extra feature 2"],
                                 id_col="id")

@op_torch
class TestTSPipeline(TestCase):

    def setUp(self) -> None:
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources/")

    def tearDown(self) -> None:
        pass

    @op_inference
    def test_seq2seq_tsppl_support_dataloader(self):
        # load
        tsppl_seq2seq = TSPipeline.load(
            os.path.join(self.resource_path, "tsppl_ckpt/s2s_tsppl_ckpt"))
        tsppl_seq2seq.fit(data=train_data_creator,
                          validation_data=valid_data_creator,
                          epochs=2,
                          batch_size=128)
        assert tsppl_seq2seq._best_config['batch_size'] == 128
        config = tsppl_seq2seq._best_config
        # predict
        yhat = tsppl_seq2seq.predict(valid_data_creator, batch_size=16)
        assert yhat.shape == (1000,
                              config['future_seq_len'],
                              config['input_feature_num'])
        assert tsppl_seq2seq._best_config['batch_size'] == 16
        yhat = tsppl_seq2seq.predict_with_onnx(valid_data_creator, batch_size=64)
        assert yhat.shape == (1000,
                              config['future_seq_len'],
                              config['input_feature_num'])
        assert tsppl_seq2seq._best_config['batch_size'] == 64

        # evaluate
        _, smape = tsppl_seq2seq.evaluate(valid_data_creator,
                                          metrics=['mse', 'smape'],
                                          batch_size=16)
        assert tsppl_seq2seq._best_config['batch_size'] == 16
        assert smape < 100.0
        _, smape = tsppl_seq2seq.evaluate_with_onnx(valid_data_creator,
                                                    metrics=['mse', 'smape'],
                                                    batch_size=64)
        assert tsppl_seq2seq._best_config['batch_size'] == 64
        assert smape < 100.0

        # evaluate with customized metrics
        from torchmetrics.functional import mean_squared_error
        def customized_metric(y_true, y_pred):
            return mean_squared_error(torch.from_numpy(y_pred),
                                      torch.from_numpy(y_true)).numpy()
        tsppl_seq2seq.evaluate(valid_data_creator,
                               metrics=[customized_metric],
                               batch_size=16)
        assert tsppl_seq2seq._best_config['batch_size'] == 16

        with pytest.raises(RuntimeError):
            tsppl_seq2seq.predict(torch.randn(1000,
                                  config['past_seq_len'],
                                  config['input_feature_num']))
        with pytest.raises(RuntimeError):
            tsppl_seq2seq.evaluate(torch.randn(1000,
                                   config['past_seq_len'],
                                   config['input_feature_num']))

    def test_tcn_tsppl_support_dataloader(self):
        # load
        tsppl_tcn = TSPipeline.load(
            os.path.join(self.resource_path, "tsppl_ckpt/tcn_tsppl_ckpt"))
        tsppl_tcn.fit(data=train_data_creator,
                      validation_data=valid_data_creator,
                      epochs=2,
                      batch_size=128)
        assert tsppl_tcn._best_config['batch_size'] == 128
        config = tsppl_tcn._best_config
        yhat = tsppl_tcn.predict(data=valid_data_creator, batch_size=16)
        assert tsppl_tcn._best_config['batch_size'] == 16
        assert yhat.shape == (1000,
                              config['future_seq_len'],
                              config['output_feature_num'])

        _, smape = tsppl_tcn.evaluate(data=valid_data_creator,
                                      metrics=['mse', 'smape'],
                                      batch_size=64)
        assert tsppl_tcn._best_config['batch_size'] == 64
        assert smape < 100.0

    def test_lstm_tsppl_support_dataloader(self):
        # load
        tsppl_lstm = TSPipeline.load(
            os.path.join(self.resource_path, "tsppl_ckpt/lstm_tsppl_ckpt"))
        tsppl_lstm.fit(data=train_data_creator,
                       validation_data=valid_data_creator,
                       epochs=2,
                       batch_size=128)
        assert tsppl_lstm._best_config['batch_size'] == 128
        config = tsppl_lstm._best_config
        yhat = tsppl_lstm.predict(data=valid_data_creator, batch_size=16)
        assert tsppl_lstm._best_config['batch_size'] == 16
        assert yhat.shape == (1000,
                              config['future_seq_len'],
                              config['output_feature_num'])
        _, smape = tsppl_lstm.evaluate(data=valid_data_creator,
                              metrics=['mse', 'smape'],
                              batch_size=64)
        assert tsppl_lstm._best_config['batch_size'] == 64
        assert smape < 100.0

    def test_tsppl_mixed_data_type_usage(self):
        # This ckpt is generated by fit on a data creator
        tsppl_lstm = TSPipeline.load(
            os.path.join(self.resource_path, "tsppl_ckpt/lstm_tsppl_ckpt"))
        with pytest.raises(RuntimeError):
            yhat = tsppl_lstm.predict(data=get_test_tsdataset(), batch_size=16)

    @op_inference
    def test_tsppl_quantize_data_creator(self):
        # s2s not support quantize
        with pytest.raises(RuntimeError):
            tsppl_s2s = TSPipeline.load(os.path.join(self.resource_path,
                                                     "tsppl_ckpt/s2s_tsppl_ckpt"))
            tsppl_s2s.quantize(calib_data=train_data_creator,
                               metric=['smape'],
                               framework=['pytorch_fx', 'onnxrt_qlinearops'])
            del tsppl_s2s

        tsppl_lstm = TSPipeline.load(os.path.join(self.resource_path,
                                                  "tsppl_ckpt/lstm_tsppl_ckpt"))
        assert tsppl_lstm._best_config['batch_size'] == 32

        yhat = tsppl_lstm.predict(valid_data_creator, batch_size=64)
        smape = tsppl_lstm.evaluate(valid_data_creator,
                                    metrics=['smape'])

        tsppl_lstm.quantize(calib_data=train_data_creator,
                            metric='mae',
                            framework=['pytorch_fx', 'onnxrt_qlinearops'])
        # only quantize
        q_yhat = tsppl_lstm.predict(valid_data_creator, batch_size=32, quantize=True)
        q_smape = tsppl_lstm.evaluate(valid_data_creator,
                                      metrics=['smape'],
                                      batch_size=128,
                                      quantize=True)

        # quantize_onnx
        q_onnx_yhat = tsppl_lstm.predict_with_onnx(valid_data_creator,
                                                   batch_size=64,
                                                   quantize=True)

        q_onnx_smape = tsppl_lstm.evaluate_with_onnx(valid_data_creator,
                                                     metrics=['smape'],
                                                     batch_size=64,
                                                     quantize=True)
        assert tsppl_lstm._best_config['batch_size'] == 64

        tsppl_lstm.fit(train_data_creator, epochs=2, batch_size=64)

        assert q_yhat.shape == yhat.shape == q_onnx_yhat.shape
        assert all([np.mean(q_smape)<100., np.mean(q_onnx_smape)<100., np.mean(smape)<100.])

    @op_inference
    def test_tsppl_quantize_input_data(self):
        tsppl_tcn = TSPipeline.load(os.path.join(self.resource_path,
                                                 "tsppl_ckpt/tcn_tsppl_ckpt"))
        config = tsppl_tcn._best_config
        # tuple input
        calib_x = np.random.randn(1000, config['past_seq_len'],
                                  config['input_feature_num']).astype(np.float32)
        calib_y = np.random.randn(1000, config['future_seq_len'],
                                  config['output_feature_num']).astype(np.float32)
        tsppl_tcn.quantize(calib_data=(calib_x, calib_y))

        with pytest.raises(RuntimeError):
            tsppl_tcn.quantize(calib_data=(calib_x, calib_y),
                               metric='smape',
                               approach='dynamic')
        with pytest.raises(RuntimeError):
            tsppl_tcn.quantize(calib_data=None,
                               metric='smape',
                               approach='static')

    @op_inference
    def test_tsppl_quantize_public_dataset(self):
        tsppl_tcn = TSPipeline.load(os.path.join(self.resource_path,
                                                 "tsppl_ckpt/tcn_tsppl_ckpt"))
        train_tsdata = get_test_tsdataset()
        test_tsdata = get_test_tsdataset()
        train_tsdata.roll(lookback=10, horizon=2)
        test_tsdata.roll(lookback=10, horizon=2)
        # mixed data
        tsppl_tcn._best_config.update({'selected_features': []})
        tsppl_tcn.quantize(calib_data=train_tsdata,
                           metric='smape')
        yhat = tsppl_tcn.predict(train_tsdata)


if __name__ == "__main__":
    pytest.main([__file__])

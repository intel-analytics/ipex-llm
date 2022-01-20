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

import tempfile
from unittest import TestCase
import pytest
import torch
import os

from torch.utils.data import TensorDataset, DataLoader
from bigdl.chronos.autots import TSPipeline


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


class TestTSPipeline(TestCase):

    def setUp(self) -> None:
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources/")

    def tearDown(self) -> None:
        pass

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
        assert smape < 2.0
        _, smape = tsppl_seq2seq.evaluate_with_onnx(valid_data_creator,
                                                    metrics=['mse', 'smape'],
                                                    batch_size=64)
        assert tsppl_seq2seq._best_config['batch_size'] == 64
        assert smape < 2.0

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
        assert smape < 2.0

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
        assert smape < 2.0

    def test_nbeats_tsppl_support_dataloader(self):
        tsppl_nbeats = TSPipeline.load(
        os.path.join(self.resource_path, "tsppl_ckpt/lstm_tsppl_ckpt"))
        tsppl_nbeats.fit(data=train_data_creator,
                         validation_data=valid_data_creator,
                         epochs=2,
                         batch_size=128)
        assert tsppl_nbeats._best_config['batch_size'] == 128
        config = tsppl_nbeats._best_config
        yhat = tsppl_nbeats.predict(data=valid_data_creator, batch_size=16)
        assert tsppl_nbeats._best_config['batch_size'] == 16
        assert yhat.shape == (1000,
                              config['future_seq_len'],
                              config['output_feature_num'])
        _, smape = tsppl_nbeats.evaluate(data=valid_data_creator,
                                         metrics=['mse', 'smape'],
                                         batch_size=64)
        assert tsppl_nbeats._best_config['batch_size'] == 64
        assert smape < 2.0

if __name__ == "__main__":
    pytest.main([__file__])

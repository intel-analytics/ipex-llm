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

from torch.utils.data import TensorDataset, DataLoader
from bigdl.chronos.autots import AutoTSEstimator, TSPipeline
from bigdl.orca.common import init_orca_context, stop_orca_context


def train_data_creator(config):
    return DataLoader(TensorDataset(torch.randn(10000,
                                                config.get('past_seq_len', 10),
                                                config.get('input_feature_num', 2)),
                                    torch.randn(10000,
                                                config.get('future_seq_len', 2),
                                                config.get('output_feature_num', 2))),
                      batch_size=config.get('batch_size', 32), shuffle=True)

def valid_data_creator(config):
    return DataLoader(TensorDataset(torch.randn(10000,
                                                config.get('past_seq_len', 10),
                                                config.get('input_feature_num', 2)),
                                    torch.randn(10000,
                                                config.get('future_seq_len', 2),
                                                config.get('output_feature_num', 2))),
                      batch_size=config.get('batch_size', 32), shuffle=False)


class TestTSPipeline(TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_seq2seq_tsppl_seq2seq_support_dataloader(self):
        tmp_seq2seq_dir = tempfile.TemporaryDirectory()
        init_orca_context(cores=4, memory="4g")
        autots = AutoTSEstimator(model="seq2seq",
                                 search_space="minimal",
                                 input_feature_num=2,
                                 output_target_num=2,
                                 past_seq_len=10,
                                 future_seq_len=2)
        tsppl_seq2seq = autots.fit(data=train_data_creator({}),
                                   validation_data=valid_data_creator({}),
                                   epochs=2,
                                   batch_size=32)
        tsppl_seq2seq.save(tmp_seq2seq_dir.name)
        del tsppl_seq2seq
        stop_orca_context()

        # load
        tsppl_seq2seq = TSPipeline.load(tmp_seq2seq_dir.name)
        tsppl_seq2seq.fit(data=train_data_creator,
                          validation_data=valid_data_creator,
                          epochs=2,
                          batch_size=128)
        assert tsppl_seq2seq._best_config['batch_size'] == 128
        config = tsppl_seq2seq._best_config
        # predict
        yhat = tsppl_seq2seq.predict(valid_data_creator, batch_size=16)
        assert yhat.shape == (10000,
                              config['future_seq_len'],
                              config['input_feature_num'])
        assert tsppl_seq2seq._best_config['batch_size'] == 16
        yhat = tsppl_seq2seq.predict_with_onnx(valid_data_creator, batch_size=64)
        assert yhat.shape == (10000,
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
            tsppl_seq2seq.predict(torch.randn(10000,
                                  config['past_seq_len'],
                                  config['input_feature_num']))
        with pytest.raises(RuntimeError):
            tsppl_seq2seq.evaluate(torch.randn(10000,
                                   config['past_seq_len'],
                                   config['input_feature_num']))

    def test_tcn_tsppl_seq2seq_support_dataloader(self):
        tmp_tcn_dir = tempfile.TemporaryDirectory()
        init_orca_context(cores=4, memory="4g")
        autots = AutoTSEstimator(model="tcn",
                                 search_space="minimal",
                                 input_feature_num=2,
                                 output_target_num=2,
                                 past_seq_len=10,
                                 future_seq_len=2)
        tsppl_tcn = autots.fit(data=train_data_creator({}),
                               validation_data=valid_data_creator({}),
                               epochs=2,
                               batch_size=32)
        tsppl_tcn.save(tmp_tcn_dir.name)
        del tsppl_tcn
        stop_orca_context()

        # load
        tsppl_tcn = TSPipeline.load(tmp_tcn_dir.name)
        tsppl_tcn.fit(data=train_data_creator,
                      validation_data=valid_data_creator,
                      epochs=2,
                      batch_size=128)
        assert tsppl_tcn._best_config['batch_size'] == 128
        config = tsppl_tcn._best_config
        yhat = tsppl_tcn.predict(data=valid_data_creator, batch_size=16)
        assert tsppl_tcn._best_config['batch_size'] == 16
        assert yhat.shape == (10000,
                              config['future_seq_len'],
                              config['output_feature_num'])

        _, smape = tsppl_tcn.evaluate(data=valid_data_creator,
                                      metrics=['mse', 'smape'],
                                      batch_size=64)
        assert tsppl_tcn._best_config['batch_size'] == 64
        assert smape < 2.0

if __name__ == "__main__":
    pytest.main([__file__])

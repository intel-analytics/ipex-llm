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
import numpy
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

    def test_seq2seq_tsppl_support_dataloader(self):
        tmp_dir = tempfile.TemporaryDirectory()
        init_orca_context(cores=4, memory="4g")
        autots = AutoTSEstimator(model="seq2seq",
                                 search_space="minimal",
                                 input_feature_num=2,
                                 output_target_num=2,
                                 past_seq_len=10,
                                 future_seq_len=2)
        tsppl = autots.fit(data=train_data_creator({}),
                           validation_data=valid_data_creator({}),
                           epochs=2)
        tsppl.save(tmp_dir.name)
        del tsppl
        stop_orca_context()

        tsppl = TSPipeline.load(tmp_dir.name)
        tsppl.fit(data=train_data_creator,
                  validation_data=valid_data_creator,
                  epochs=2)
        config = tsppl._best_config
        # predict
        yhat = tsppl.predict(valid_data_creator)
        assert tuple(yhat.shape) == (10000,
                                     config['future_seq_len'],
                                     config['input_feature_num'])
        yhat = tsppl.predict_with_onnx(valid_data_creator)
        assert tuple(yhat.shape) == (10000,
                                     config['future_seq_len'],
                                     config['input_feature_num'])

        # evaluate
        mse, smape = tsppl.evaluate(valid_data_creator, metrics=['mse', 'smape'])
        assert isinstance(mse, numpy.ndarray) and smape < 2.0
        mse, smape = tsppl.evaluate_with_onnx(valid_data_creator, metrics=['mse', 'smape'])
        assert isinstance(mse, numpy.ndarray) and smape < 2.0

        with pytest.raises(RuntimeError):
            tsppl.predict(torch.randn(10000,
                                      config['past_seq_len'],
                                      config['input_feature_num']))
        with pytest.raises(RuntimeError):
            tsppl.evaluate(torch.randn(10000,
                                       config['past_seq_len'],
                                       config['input_feature_num']))

if __name__ == "__main__":
    pytest.main([__file__])

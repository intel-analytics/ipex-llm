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
from bigdl.chronos.utils import LazyImport

class TestChronosModelPytorch(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tcn_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import TCNForecaster
        assert isinstance(TCNForecaster, LazyImport)

        tcn = TCNForecaster(input_feature_num=1,
                            output_feature_num=1,
                            past_seq_len=48,
                            future_seq_len=5)
        from bigdl.chronos.forecaster.tcn_forecaster import TCNForecaster
        assert not isinstance(TCNForecaster, LazyImport)
        assert isinstance(tcn, TCNForecaster)

    def test_lstm_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import LSTMForecaster
        assert isinstance(LSTMForecaster, LazyImport)

        lstm = LSTMForecaster(input_feature_num=1,
                              output_feature_num=1,
                              past_seq_len=48)
        from bigdl.chronos.forecaster.lstm_forecaster import LSTMForecaster
        assert not isinstance(LSTMForecaster, LazyImport)
        assert isinstance(lstm, LSTMForecaster)
                                      

    def test_nbeats_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import NBeatsForecaster
        assert isinstance(NBeatsForecaster, LazyImport)

        nbeats = NBeatsForecaster(past_seq_len=48,
                                  future_seq_len=5)
        from bigdl.chronos.forecaster.nbeats_forecaster import NBeatsForecaster
        assert not isinstance(NBeatsForecaster, LazyImport)
        assert isinstance(nbeats, NBeatsForecaster)

    def test_s2s_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import Seq2SeqForecaster
        assert isinstance(Seq2SeqForecaster, LazyImport)

        from bigdl.chronos.forecaster.seq2seq_forecaster import Seq2SeqForecaster
        s2s = Seq2SeqForecaster(past_seq_len=48,
                                future_seq_len=5,
                                input_feature_num=1,
                                output_feature_num=1)
        assert not isinstance(Seq2SeqForecaster, LazyImport)
        assert isinstance(s2s, Seq2SeqForecaster)

    def test_autoformer_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import AutoformerForecaster
        assert isinstance(AutoformerForecaster, LazyImport)

        from bigdl.chronos.forecaster.autoformer_forecaster import AutoformerForecaster
        autoformer = AutoformerForecaster(past_seq_len=48,
                                          future_seq_len=5,
                                          input_feature_num=1,
                                          output_feature_num=1,
                                          label_len=12,
                                          freq="s")
        assert not isinstance(autoformer, LazyImport)
        assert isinstance(autoformer, AutoformerForecaster)

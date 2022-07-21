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

class TestChronosModelPytorch(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tcn_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import TCNForecaster

    def test_lstm_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import LSTMForecaster

    def test_nbeats_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import NBeatsForecaster

    def test_s2s_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import Seq2SeqForecaster

    def test_autoformer_forecaster_import(self):
        from bigdl.chronos.forecaster.pytorch import AutoformerForecaster

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

import importlib
from bigdl.chronos.utils import LazyImport

tf_spec = bool(importlib.util.find_spec('tensorflow'))
orca_available = bool(importlib.util.find_spec('bigdl.orca'))

PREFIXNAME_TF2 = 'bigdl.chronos.forecaster.tf.'

if tf_spec:
    LSTMForecaster = LazyImport(PREFIXNAME_TF2+'lstm_forecaster.LSTMForecaster')
    Seq2SeqForecaster = LazyImport(PREFIXNAME_TF2+'seq2seq_forecaster.Seq2SeqForecaster')
    TCNForecaster = LazyImport(PREFIXNAME_TF2+'tcn_forecaster.TCNForecaster')
    if orca_available:
        MTNetForecaster = LazyImport(PREFIXNAME_TF2+'metnet_forecaster.MTNetForecaster')

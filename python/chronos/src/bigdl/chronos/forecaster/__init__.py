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

import logging
import importlib
from bigdl.chronos.utils import LazyImport
# unset the KMP_INIT_AT_FORK
# which will cause significant slow down in multiprocessing training
import os
os.unsetenv('KMP_INIT_AT_FORK')

class Disablelogging:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, *args, **opts):
        logging.disable(logging.NOTSET)

# dependencies check
torch_available = bool(importlib.util.find_spec('torch'))
tf_available = False
try:
    tf = LazyImport('tensorflow')
    tf_available = tf.__version__ > "2.0.0"
except:
    pass

# Avoid printing redundant message
prophet_available = False
try:
    with Disablelogging():
        import prophet
    prophet_available = True
except:
    pass

arima_available = bool(importlib.util.find_spec('pmdarima'))
orca_available = bool(importlib.util.find_spec('bigdl.orca'))

# import forecasters
PREFIXNAME = 'bigdl.chronos.forecaster.'
if torch_available:
    LSTMForecaster = LazyImport(PREFIXNAME+'lstm_forecaster.LSTMForecaster')
    TCNForecaster = LazyImport(PREFIXNAME+'tcn_forecaster.TCNForecaster')
    Seq2SeqForecaster = LazyImport(PREFIXNAME+'seq2seq_forecaster.Seq2SeqForecaster')
    NBeatsForecaster = LazyImport(PREFIXNAME+'nbeats_forecaster.NBeatsForecaster')
    AutoformerForecaster = LazyImport(PREFIXNAME+'autoformer_forecaster.AutoformerForecaster')
    if orca_available:
        TCMFForecaster = LazyImport(PREFIXNAME+'tcmf_forecaster.TCMFForecaster')
if tf_available and orca_available:
    MTNetForecaster = LazyImport(PREFIXNAME+'tf.mtnet_forecaster.MTNetForecaster')
if prophet_available:
    ProphetForecaster = LazyImport(PREFIXNAME+'prophet_forecaster.ProphetForecaster')
if arima_available:
    ARIMAForecaster = LazyImport(PREFIXNAME+'arima_forecaster.ARIMAForecaster')

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

import warnings
import logging
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
torch_available = False
tf_available = False
prophet_available = False
arima_available = False
orca_available = False
try:
    import torch
    torch_available = True
except:
    warnings.warn("Please install `torch` to use forecasters, including TCMFForecaster, "
                  "TCNForecaster, LSTMForecaster, Seq2SeqForecaster.")
try:
    import tensorflow as tf
    assert tf.__version__ > "2.0.0"
    tf_available = True
except:
    warnings.warn("Please install `tensorflow>2.0.0` to use MTNetForecaster.")
try:
    with Disablelogging():
        import prophet
    prophet_available = True
except:
    warnings.warn("Please install `prophet` to use ProphetForecaster.")
try:
    import pmdarima
    arima_available = True
except:
    warnings.warn("Please install `pmdarima` to use ARIMAForecaster.")
try:
    import bigdl.orca
    orca_available = True
except:
    warnings.warn("Please install `bigdl-orca` to use full collection of forecasters.")

# import forecasters
PREFIX_PATH = 'bigdl.chronos.forecaster'
if torch_available:
    LSTMForecaster = LazyImport('..LSTMForecaster', pkg=PREFIX_PATH+'.lstm_forecaster')
    TCNForecaster = LazyImport('..TCNForecaster', pkg=PREFIX_PATH+'.tcn_forecaster')
    Seq2SeqForecaster = LazyImport('..Seq2SeqForecaster', pkg=PREFIX_PATH+'.seq2seq_forecaster')
    NBeatsForecaster = LazyImport('..NBeatsForecaster', pkg=PREFIX_PATH+'.nbeats_forecaster')
if tf_available and orca_available:
    from .tf.mtnet_forecaster import MTNetForecaster
    TCMFForecaster = LazyImport('..TCMFForecaster', pkg=PREFIX_PATH+'.tcmf_forecaster')
if tf_available:
    MTNetForecaster = LazyImport('..MTNetForecaster', pkg=PREFIX_PATH+'.tf.mtnet_forecaster')
if prophet_available:
    ProphetForecaster = LazyImport('..ProphetForecaster', pkg=PREFIX_PATH+'.ProphetForecaster')
if arima_available:
    ARIMAForecaster = LazyImport('..ARIMAForecaster', pkg=PREFIX_PATH+'.ARIMAForecaster')

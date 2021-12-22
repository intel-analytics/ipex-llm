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


# dependencies check
torch_available = False
tf_available = False
prophet_available = False
arima_available = False
try:
    import torch
    torch_available = True
except:
    warnings.warn(r"Please install `torch` to use forecasters, including tcmf\tcn\seq2seq\lstm.")
try:
    import tensorflow as tf
    assert tf.__version__ < "2.0.0"
    tf_available = True
except:
    warnings.warn("Please install `tensorflow<2.0.0` to use mtnet_forecaster.")
try:
    import prophet
    prophet_available = True
except:
    warnings.warn("Please install `prophet` to use prophet_forecaster.")
try:
    import pmdarima
    arima_available = True
except:
    warnings.warn("Please install `pmdarima` to use arima_forecaster.")

# import forecasters
if torch_available:
    from .lstm_forecaster import LSTMForecaster
    from .tcn_forecaster import TCNForecaster
    from .seq2seq_forecaster import Seq2SeqForecaster
    from .tcmf_forecaster import TCMFForecaster
if tf_available:
    from .mtnet_forecaster import MTNetForecaster
if prophet_available:
    from .prophet_forecaster import ProphetForecaster
if arima_available:
    from .arima_forecaster import ARIMAForecaster

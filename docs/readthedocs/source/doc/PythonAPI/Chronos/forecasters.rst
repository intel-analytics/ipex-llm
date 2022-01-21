Forecasters
=====================

LSTMForecaster
----------------------------------------

:strong:`Please refer to` `BasePytorchForecaster <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#basepytorchforecaster>`__ :strong:`for other methods other than initialization`.

Long short-term memory(LSTM) is a special type of recurrent neural network(RNN). We implement the basic version of LSTM - VanillaLSTM for this forecaster for time-series forecasting task. It has two LSTM layers, two dropout layer and a dense layer.

For the detailed algorithm description, please refer to `here <https://github.com/intel-analytics/BigDL/blob/branch-2.0/docs/docs/Chronos/Algorithm/LSTMAlgorithm.md>`__.

.. automodule:: bigdl.chronos.forecaster.lstm_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


Seq2SeqForecaster
-------------------------------------------

:strong:`Please refer to` `BasePytorchForecaster <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#basepytorchforecaster>`__ :strong:`for other methods other than initialization`.

Seq2SeqForecaster wraps a sequence to sequence model based on LSTM, and is suitable for multivariant & multistep time series forecasting.

.. automodule:: bigdl.chronos.forecaster.seq2seq_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


TCNForecaster
----------------------------------------

:strong:`Please refer to` `BasePytorchForecaster <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#basepytorchforecaster>`__ :strong:`for other methods other than initialization`.

Temporal Convolutional Networks (TCN) is a neural network that use convolutional architecture rather than recurrent networks. It supports multi-step and multi-variant cases. Causal Convolutions enables large scale parallel computing which makes TCN has less inference time than RNN based model such as LSTM.

.. automodule:: bigdl.chronos.forecaster.tcn_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


NBeatsForecaster
----------------------------------------

:strong:`Please refer to` `BasePytorchForecaster <https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Chronos/forecasters.html#basepytorchforecaster>`__ :strong:`for other methods other than initialization`.

Neural basis expansion analysis for interpretable time series forecasting (N-BEATS) is a deep neural architecture based on backward and forward residual links and a very deep stack of fully-connected layers. Nbeats can solve univariate time series point forecasting problems, being interpretable, and fast to train.

.. automodule:: bigdl.chronos.forecaster.nbeats_forecaster
    :members:
    :undoc-membears:
    :show-inheritance:


TCMFForecaster
----------------------------------------

Chronos TCMFForecaster provides an efficient way to forecast high dimensional time series.

TCMFForecaster is based on DeepGLO algorithm, which is a deep forecasting model which thinks globally and acts locally.
You can refer to `the deepglo paper <https://arxiv.org/abs/1905.03806>`__ for more details.

TCMFForecaster supports distributed training and inference. It is based on Orca PyTorch Estimator, which is an estimator to do PyTorch training/evaluation/prediction on Spark in a distributed fashion. Also you can choose to enable distributed training and inference or not.

**Remarks**:

* You can refer to `TCMFForecaster installation <https://github.com/intel-analytics/BigDL/blob/branch-2.0/docs/docs/Chronos/tutorials/TCMFForecaster.md#step-0-prepare-environment>`__ to install required packages.
* Your operating system (OS) is required to be one of the following 64-bit systems: **Ubuntu 16.04 or later** and **macOS 10.12.6 or later**.

.. automodule:: bigdl.chronos.forecaster.tcmf_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


MTNetForecaster
----------------------------------------

MTNet is a memory-network based solution for multivariate time-series forecasting. In a specific task of multivariate time-series forecasting, we have several variables observed in time series and we want to forecast some or all of the variables' value in a future time stamp.

MTNet is proposed by paper `A Memory-Network Based Solution for Multivariate Time-Series Forecasting <https://arxiv.org/abs/1809.02105>`__. MTNetForecaster is derived from tfpark.KerasMode, and can use all methods of KerasModel. Refer to `tfpark.KerasModel API Doc <https://github.com/intel-analytics/BigDL/blob/branch-2.0/docs/docs/APIGuide/TFPark/model.md>`__ for details.

For the detailed algorithm description, please refer to `here <https://github.com/intel-analytics/BigDL/blob/branch-2.0/docs/docs/Chronos/Algorithm/MTNetAlgorithm.md>`__.

.. automodule:: bigdl.chronos.forecaster.mtnet_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


ARIMAForecaster
----------------------------------------

AutoRegressive Integrated Moving Average (ARIMA) is a class of statistical models for analyzing and forecasting time series data. It consists of 3 components: AR (AutoRegressive), I (Integrated) and MA (Moving Average). In ARIMAForecaster we use the SARIMA model (Seasonal ARIMA), which is an extension of ARIMA that additionally supports the direct modeling of the seasonal component of the time series.

.. automodule:: bigdl.chronos.forecaster.arima_forecaster
    :members:
    :undoc-members:
    :show-inheritance:


ProphetForecaster
----------------------------------------

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

For the detailed algorithm description, please refer to `here <https://github.com/facebook/prophet>`__.

.. automodule:: bigdl.chronos.forecaster.prophet_forecaster
    :members:
    :undoc-members:
    :show-inheritance:
    
    
TFParkForecaster
----------------------------------------

.. automodule:: bigdl.chronos.forecaster.tfpark_forecaster
    :members:
    :undoc-members:
    :show-inheritance:

BasePytorchForecaster
----------------------------------------

.. autoclass:: bigdl.chronos.forecaster.base_forecaster.BasePytorchForecaster
    :members:
    :show-inheritance:
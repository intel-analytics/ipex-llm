# MTNetForecaster

## Introduction

MTNet is a memory-network based solution for multivariate time-series forecasting. In a specific task of multivariate time-series forecasting, we have several variables observed in time series and we want to forecast some or all of the variables' value in a future time stamp.

MTNet is proposed by paper [A Memory-Network Based Solution for Multivariate Time-Series Forecasting](https://arxiv.org/abs/1809.02105). MTNetForecaster is derived from tfpark.KerasMode, and can use all methods of KerasModel. Refer to [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md) for details.

For the detailed algorithm description, please refer to [here](../Algorithm/MTNetAlgorithm.md).

## Method

### Arguments

- **`target_dim`**: Specify the number of variables we want to forecast. i.e. the the dimension of model output feature. This value defaults to 1.
- **`feature_dim`**: Specify the number of variables we have in the input data. i.e. the the dimension of model input feature. This value defaults to 1.
- **`long_series_num`**: Specify the number of long-term historical data series. This value defaults to 1. Typically, as stated in the [paper](https://arxiv.org/abs/1809.02105), the value is set to 7.
- **`series_length`**: Specify the length of long-term historical data series, which is equal to the length of short-term data series. This value defaults to 1. The value should be larger or equal to 1.
- **`ar_window_size`**: Specify the auto regression window size in MTNet. This value defaults to 1. Since the parameter is along the time dimension, the value should be smaller or equal to `series_length`.
- **`cnn_height`**: Specify convolutional layer filter height in MTNet's encoder. This value defaults to 1. Since the parameter is along the time dimension, the value should be smaller or equal to `series_length`.
- **`cnn_hid_size`**: Specify the convolutional layer filter number in MTNet's encoder. This value defaults to 32. Typically, as stated in the [paper](https://arxiv.org/abs/1809.02105), the value is grid searched in {32, 50, 100}.
- **`rnn_hid_size`**: Specify the the hidden RNN layers unit number in MTNet's encoder. This value defaults to [16, 32] as a stacked RNN.  Typically, as stated in the [paper](https://arxiv.org/abs/1809.02105), the value is grid searched in {32, 50, 100} for each layer. This parameter should be set as a list.
- **`lr`**: Specify the learning rate. This value defaults to 0.001.
- **`loss`**: Specify the target function you want to optimize on. This value defaults to MAE. 
- **`metric`**: Specify the metric for validation and evaluation. This value defaults to MSE.
- **`cnn_dropout`**: Specify the dropout close possibility for CNN in encoder. This value defaults to 0.2, as stated in the [paper](https://arxiv.org/abs/1809.02105).
- **`rnn_dropout`**: Specify the dropout close possibility for RNN in encoder. This value defaults to 0.2, as stated in the [paper](https://arxiv.org/abs/1809.02105).
- **`uncertainty`**: Specify whether the forecaster can perform the calculation of uncertainty.

### \__init__

```python
MTNetForecaster(target_dim=1,
                 feature_dim=1,
                 long_series_num=1,
                 series_length=1,
                 ar_window_size=1,
                 cnn_height=1,
                 cnn_hid_size=32,
                 rnn_hid_sizes=[16, 32],
                 lr=0.001,
                 loss="mae",
                 cnn_dropout=0.2,
                 rnn_dropout=0.2,
                 metric="mean_squared_error",
                 uncertainty: bool = False,
                 )

```

### fit, evaluate, predict

Refer to **fit**, **evaluate**, **predict** defined in [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md)

## Reference

Yen-YuChang, Fan-YunSun, Yueh-HuaWu, Shou-DeLin,  [A Memory-Network Based Solution for Multivariate Time-Series Forecasting](https://arxiv.org/abs/1809.02105). 


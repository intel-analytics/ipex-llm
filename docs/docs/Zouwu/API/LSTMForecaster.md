# LSTMForecaster

## Introduction

Long short-term memory(LSTM) is a special type of recurrent neural network(RNN). We implement the basic version of LSTM - VanillaLSTM for this forecaster for time-series forecasting task. It has two LSTM layers, two dropout layer and a dense layer. LSTMForecaster is derived from tfpark.KerasMode, and can use all methods of KerasModel. Refer to [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md) for details.

For the detailed algorithm description, please refer to [here](../Algorithm/LSTMAlgorithm.md).

## Method

### Arguments

- **`target_dim`**: Specify the number of variables we want to forecast. i.e. the the dimension of model output feature. This value defaults to 1.
- **`feature_dim`**: Specify the number of variables we have in the input data. i.e. the the dimension of model input feature. This value defaults to 1.
- **`lstm_units`**: Specify the dimensionality of the output space for LSTM layers. This value defaults to (16, 8). 
- **`dropouts`**: Specify the fraction of the input units to drop for dropout layers. This value defaults to 0.2. Note that The same dropout rate will be set to all
            layers if dropouts is one float value while lstm_units has multiple elements.
- **`metric`**: Specify the metric for validation and evaluation. This value defaults to MSE.
- **`lr`**: Specify the learning rate. This value defaults to 0.001.
- **`loss`**: Specify the target function you want to optimize on. This value defaults to MSE.
- **`optimizer`**: Specify the optimizer used for training. This value defaults to Adam.

### \__init__

```python
LSTMForecaster(target_dim=1,
               feature_dim=1,
               lstm_units=(16, 8),
               dropouts=0.2,
               metric="mean_squared_error",
               lr=0.001,
               loss="mse",
               optimizer="Adam",
               )
```

### fit, evaluate, predict

Refer to **fit**, **evaluate**, **predict** defined in [tfpark.KerasModel API Doc](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/APIGuide/TFPark/model.md)

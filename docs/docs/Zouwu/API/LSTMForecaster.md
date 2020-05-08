# LSTMForecaster

LSTMForecaster is derived from tfpark.KerasMode, and can use all methods of KerasModel. Refer to [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md) for details.

## Methods

### \_\_init\_\_

```python
LSTMForecaster(target_dim=1,
               feature_dim=1,
               lstm_1_units=16,
               dropout_1=0.2,
               lstm_2_units=8,
               dropout_2=0.2,
               metric="mean_squared_error",
               lr=0.001,
               uncertainty: bool = False
            )
```

#### Arguments
* **target_dim**: dimension of model output
* **feature_dim**: dimension of input feature
* **lstm_1_units** num of units for the 1st LSTM layer
* **dropout_1**: p for the 1st dropout layer
* **lstm_2_units**: num of units for the 2nd LSTM layer
* **dropout_2**: p for the 2nd dropout layer
* **metric**: the metric for validation and evaluation
* **lr**: learning rate
* **uncertainty**: whether to return uncertainty

### fit, evaluate, predict

Refer to **fit**, **evaluate**, **predict** defined in [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md)

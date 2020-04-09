# MTNetForecaster

MTNetForecaster is derived from tfpark.KerasMode, and can use all methods of KerasModel. Refer to [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md) for details.

## Methods

### \_\_init\_\_

```python
MTNetForecaster(horizon=1,
               feature_dim=1,
               lb_long_steps=1,
               lb_long_stepsize=1,
               metric="mean_squared_error",
               uncertainty: bool = False,
            )

```

#### Arguments
* **horizon**: the steps to look forward
* **feature_dim**: the dimension of input feature
* **lb_long_steps**: the number of steps for the long-term memory series
* **lb_long_stepsize**: the stepsize for long-term memory series
* **metric**: the metric for validation and evaluation
* **uncertainty**: whether to enable calculation of uncertainty

### fit, evaluate, predict

Refer to **fit**, **evaluate**, **predict** defined in [tfpark.KerasModel API Doc](../../APIGuide/TFPark/model.md)

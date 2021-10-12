
In this guide, we will show you how to use the built-in LSTMForecaster and MTNetForecaster for time series forecasting.

The built-in LSTMForecaster and MTNetForecaster are both derived from [tfpark.KerasModels](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/). 

Refer to [network traffic notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb) for demonstration of forecasting network traffic data with Chronos built-in LSTMForecaster and MTNetForecaster.

Refer to [LSTMForecaster API](../API/LSTMForecaster.md) and [MTNetForecaster API](../API/MTNetForecaster.md) detailed explanation of all arguments for each forecast model.

---
### **Step 0: Prepare environment**
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run automated training on a yarn cluster (yarn-client mode only).
```bash
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo[automl]==0.9.0.dev0 # or above
```

### **Step 1: Create forecast model**
To start, you need to create a forecast model first. Specify **target_dim** and **feature_dim** in constructor. 

*  ```target_dim```: dimension of target output
*  ```feature_dim```: dimension of input feature


Below are some example code to create forecast models.

```python
#import forecast models
from zoo.chronos.forecaster.lstm_forecaster import LSTMForecaster
from zoo.chronos.forecaster.mtnet_forecaster import MTNetForecaster

#build a lstm forecast model
lstm_forecaster = LSTMForecaster(target_dim=1, 
                      feature_dim=4)
                      
#build a mtnet forecast model
mtnet_forecaster = MTNetForecaster(target_dim=1,
                        feature_dim=4,
                        long_series_num=1,
                        series_length=3,
                        ar_window_size=2,
                        cnn_height=2)
```
### **Step 2: Use forecast model**
Use ```forecaster.fit/evalute/predict``` in the same way as [tfpark.KerasModel](https://analytics-zoo.github.io/master/#APIGuide/TFPark/model/)

For univariant forecasting (i.e. to predict one series at a time), you can use either **LSTMForecaster** or **MTNetForecaster**. The input data shape for `fit/evaluation/predict` should match the arguments you used to create the forecaster. Specifically:

* **X** shape should be ```(num of samples, lookback, feature_dim)```
* **Y** shape should be ```(num of samples, target_dim)```
* Where, ```feature_dim``` is the number of features as specified in Forecaster constructors. ```lookback``` is the number of time steps you want to look back in history. ```target_dim``` is the number of series to forecast at the same time as specified in Forecaster constructors and should be 1 here. If you want to do multi-step forecasting and use the second dimension as no. of steps to look forward, you won't get error but the performance may be uncertain and we don't recommend using that way.


For multivariant forecasting (i.e. to predict several series at the same time), you have to use **MTNetForecaster**. The input data shape should meet below criteria.  

* **X** shape should be ```(num of samples, lookback, feature_dim)```
* **Y** shape should be ```(num of samples, target_dim)``` 
* Where ```lookback``` should equal ```(lb_long_steps+1) * lb_long_stepsize```, where ```lb_long_steps``` and ```lb_long_stepsize``` are as specified in ```MTNetForecaster``` constructor. ```target_dim``` should equal number of series in input.

---

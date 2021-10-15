## Project Chronos: Analytics Zoo Time Series for Telco 

Time series prediction takes observations from previous time steps as input and predicts the values at future time steps. The Chronos library makes it easy to build end-to-end time series analysis by applying AutoML to extremely large-scale time series prediction, including:

* **_Use case_** - reference time series use cases in the Telco industry (such as network traffic forcasting, etc.)
* **_Model_** - built-in deep learning models for time series analysis, for example
     * LSTM - suitable for univarient forecasting
     * [MTNet](https://arxiv.org/abs/1809.02105) - suitable for both univarient and multivarient forecasting
     * TCMF - suitable for very high dimensional multivarient forecasting
* **_AutoTS_** - AutoML support for building end-to-end time series analysis pipelines (including automatic feature generation, model selection and hyperparameter tuning).

For more details about how to use, tutorials and API docs, please refer to the [Chronos User Guide](https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/chronos.html) 

---
### Reference Use Case

#### Network Traffic Forecasting
Time series forecasting has many applications in telco. Accurate forecast of telco KPIs (e.g. traffic, utilizations, user experience, etc.) for communication networks ( 2G/3G/4G/5G/wired) can help predict network failures, allocate resource, or save energy. Time series forecasting can also be used for log and metric analysis for data center IT operations for telco. Metrics to be analyzed can be hardware or VM utilizations, database metrics or servce quality indicators.

In [network traffic reference use case](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/chronos/use-case/network_traffic), we demonstrate a time series forecasting use case using a public telco dataset, i.e. the aggregated network traffic traces at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/)). In particular, we used aggregated traffic metrics (e.g. total bytes, average MBps) in the past to forecast the traffic in the furture. We demostrated two ways to do forecasting, using built-in models, and using AutoTS. 

* **Using Built-in Models**([notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb)) - In this notebook we demostrate how to use built-in "Forecaster" models to do univariant forecasting (predict only 1 series), and multivariant forecasting (predicts more than 1 series at the same time).

* **Using AutoTS**([notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb)) - In this notebook we demostrate how to use AutoTS to build a time series forcasting pipeline.

---
### Built-in models vs. AutoTS

Here we show some comparison results between manually tuned built-in models vs. AutoTS, using [network traffic forecast case](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/chronos/use-case/network_traffic) as an example.  In paritcular, we compare    
*  manually tuned built-in LSTMForecaster
*  auto-tuned LSTM-based pipeline using AutoTS (obtained out of ~100 trials)

From below results, we can see that the features selected by AutoTS make much sense in our case, and AutoTS does achieve much better accuracy results than manually tuned model.

#### Accuracy: manually tuned vs. AutoTS

|Model|Mean Squared Error (smaller is better)|Symmetric Mean Absolute Percentage Error (smaller is better)|Trained Epochs|
|--|-----|----|---|
|Manually Tuned LSTMForecaster|6312.44|8.61%|50|
|AutoTS (LSTM model)|2792.22|5.80%|50|


#### Hyper parameters: manually tuned vs. by AutoTS. 

||features|Batch size|learning rate|lstm_units*|dropout_p*|Lookback|
|--|--|--|-----|-----|-----|-----|
|Manually Tuned LSTMForecaster|year, month, week, day_of_week, hour|1024|0.001|32, 32|0.2, 0.2|84|
|AutoTS (LSTM model)|hour, is_weekday, is_awake|64|0.001|32, 64|0.2, 0.236|55|

_*_: There're 2 lstm layers and dropout in LSTM model, the hyper parameters in the table corresponds to the 1st and 2nd layer respectively. 




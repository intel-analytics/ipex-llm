
## Overview

Project Zouwu provides a reference solution that is designed and optimized for common time series applications in the Telco industry, including:
* _Use case_ - reference time series use cases in the Telco industry (such as network traffic forcasting, etc.)
* _Model_ - built-in deep learning models for time series analysis (such as LSTM and [MTNet](https://arxiv.org/abs/1809.02105))
* _AutoTS_ - AutoML support for building end-to-end time series analysis pipelines (including automatic feature generation, model selection and hyperparameter tuning).


## Requirements 

* Python 3.6 or 3.7
* PySpark 2.4.3
* Ray 0.7.0
* Tensorflow 1.15.0
* aiohttp
* setproctitle
* scikit-learn >=0.20.0
* psutil
* requests
* featuretools
* pandas
* Note that Keras is not needed to use Zouwu. But if you have Keras installed, make sure it is Keras 1.2.2. Other verisons might cause unexpected problems. 

## Forecasting

Time series forecasting has many applications in telco. Accurate forecast of telco KPIs (e.g. traffic, utilizations, user experience, etc.) for communication networks ( 2G/3G/4G/5G/wired) can help predict network failures, allocate resource, or save energy. Time series forecasting can also be used for log and metric analysis for data center IT operations for telco. Metrics to be analyzed can be hardware or VM utilizations, database metrics or servce quality indicators.

We provided a reference use case where we forecast network traffic KPI's as a demo. Refer to [Network Traffic](./use-case/network-traffic) for forecasting.
 


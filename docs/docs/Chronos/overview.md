#Project Chronos: Analytics Zoo Time Series for Telco 

Project Chronos provides a reference solution that is designed and optimized for common time series applications in the Telco industry, including:

* _Use case_ - reference time series use cases in the Telco industry (such as network traffic forcasting, etc.)
* _Model_ - built-in deep learning models for time series analysis (such as LSTM, [MTNet](https://arxiv.org/abs/1809.02105) and TCMF)
* _AutoTS_ - AutoML support for building end-to-end time series analysis pipelines (including automatic feature generation, model selection and hyperparameter tuning).


## Forecasting

Time series forecasting has many applications in telco. Accurate forecast of telco KPIs (e.g. traffic, utilizations, user experience, etc.) for communication networks ( 2G/3G/4G/5G/wired) can help predict network failures, allocate resource, or save energy. Time series forecasting can also be used for log and metric analysis for data center IT operations for telco. Metrics to be analyzed can be hardware or VM utilizations, database metrics or servce quality indicators.

We provided a reference use case where we forecast network traffic KPI's as a demo. Refer to [Network Traffic](./use-case/network-traffic) for forecasting.
 
To learn how to use built-in models, refer to tutorials (i.e. [LSTMForecaster and MTNetForcaster](tutorials/LSTMForecasterAndMTNetForecaster.md), [TCMFForecaster](tutorials/TCMFForecaster.md)) and API docs (i.e. [LSTMForecaster](./API/LSTMForecaster.md), [MTNetForecaster](./API/MTNetForecaster.md)) and [TCMFForecaster](./API/TCMFForecaster.md) for built-in models. 

To learn how to use AutoTS, refer to [AutoTS tutorial](./tutorials/Autots.md) and API docs (i.e. [AutoTSTrainer](./API/AutoTSTrainer.md) and [TSPipeline](./API/TSPipeline.md)) for automated training.



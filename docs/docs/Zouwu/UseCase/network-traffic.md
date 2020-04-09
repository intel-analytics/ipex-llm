
## Network Traffic Forecasting


Time series forecasting has many applications in telco. Accurate forecast of telco KPIs (e.g. traffic, utilizations, user experience, etc.) for communication networks ( 2G/3G/4G/5G/wired) can help predict network failures, allocate resource, or save energy. Time series forecasting can also be used for log and metric analysis for data center IT operations for telco. Metrics to be analyzed can be hardware or VM utilizations, database metrics or servce quality indicators. 

In [network traffic reference use case](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/zouwu/use-case/network_traffic), we demonstrate a time series forecasting use case using a public telco dataset, i.e. the aggregated network traffic traces at the transit link of WIDE to the upstream ISP ([dataset link](http://mawi.wide.ad.jp/~agurim/dataset/)). In particular, we used aggregated traffic metrics (e.g. total bytes, average MBps) in the past to forecast the traffic in the furture. 

---
## Using Bulit-in Models

In this [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/zouwu/use-case/network_traffic/network_traffic_model_forecasting.ipynb), we demostrate how to use built-in forecaster models to do univariant forecasting (predict only 1 series), and multivariant forecasting (predicts more than 1 series at the same time).

---
## Using AutoTS

In this [notebook](https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/zouwu/use-case/network_traffic/network_traffic_autots_forecasting.ipynb), we demostrate how to use AutoTS to build a time series forcasting pipeline. 



 


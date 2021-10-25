# Time Series Anomaly Detection Overview

Anomaly Detection detects abnormal samples in a given time series. _Chronos_ provides a set of unsupervised anomaly detectors. 

View some examples notebooks for [Datacenter AIOps](https://github.com/intel-analytics/BigDL/tree/branch-2.0/python/chronos/use-case/AIOps).

## **1. ThresholdDetector**

ThresholdDetector detects anomaly based on threshold. It can be used to detect anomaly on a given time series ([notebook](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb)), or used together with [Forecasters](#forecasting) to detect anomaly on new coming samples ([notebook](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.ipynb)). 

View [ThresholdDetector API Doc](../../PythonAPI/Chronos/anomaly_detectors.html#chronos-model-anomaly-th-detector) for more details.


## **2. AEDetector**

AEDetector detects anomaly based on the reconstruction error of an autoencoder network. 

View anomaly detection [notebook](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb) and [AEDetector API Doc](../../PythonAPI/Chronos/anomaly_detectors.html#chronos-model-anomaly-ae-detector) for more details.

## **3. DBScanDetector**

DBScanDetector uses DBSCAN clustering algortihm for anomaly detection. 

View anomaly detection [notebook](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.ipynb) and [DBScanDetector API Doc](../../PythonAPI/Chronos/anomaly_detectors.html#chronos-model-anomaly-dbscan-detector) for more details.

## Anomaly detection use case in Zouwu

---
We demonstrate how to use Zouwu to detect anomaly data  based on historical time series data. And measure the distance between predicted values and actual values. If the distance is above some threshold, we report those values as anomaly.

In the reference use case, we use the publicly available cluster trace data cluster-trace-v2018 of Alibaba Open Cluster Trace Program. ([dataset link](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz)). 
 

This use case example contains a notebook:

- **AIOps_anomaly_detect.ipynb** demonstrates how to leverage Zouwu's built-in models ie. MTNet, to do time series forecasting. 


### Install

You can refer to Zouwu installation document [here](https://analytics-zoo.github.io/master/#Zouwu/tutorials/LSTMForecasterAndMTNetForecaster/#step-0-prepare-environment).

### Prepare dataset
* run `get_data.sh` to download the full dataset. It will download the resource usage of each machine from m_1932 to m_2085. 
* run `extract_data.sh` to extract records of machine 1932. The script will extract the m_1932 with timestamps into `m_1932.csv`.



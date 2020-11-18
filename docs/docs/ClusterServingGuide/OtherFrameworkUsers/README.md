# Other Framework User Guide

This guide is for users who:

* have written local code of Tensorflow, Pytorch, OpenVINO
* have used specified data type of a specific framework, e.g. TFDataSet
* want to deploy the local code on Cluster Serving but do not know how to write client code (Cluster Serving takes Numpy Ndarray as input, other types need to transform in advance).

**If you have the above needs but fail to find the solution below, please [create issue here](https://github.com/intel-analytics/analytics-zoo/issues)

## Tensorflow
### Model
It is recommended to use savedModel format, Frozen Graph is also supported.
* Keras Model to savedModel:
* Checkpoint to Frozen Graph:
### Data
To transform following data type to Numpy Ndarray
* TFDataSet: 

## Pytorch

## OpenVINO
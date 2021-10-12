# Anomaly Detection
This is a simple example of unsupervised anomaly detection using Analytics Zoo Keras-Style API. We use RNN to predict following data values based on previous sequence (in order) and measure the distance between predicted values and actual values. If the distance is above some threshold, we report those values as anomaly.

## Requirement
* Python 3.6/3.7 (pandas 1.0+)
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)

## Install
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Download data
* Download Analytics Zoo prebuilt zip and extract into `$ANALYTICS_ZOO_HOME`. 
* Run `$ANALYTICS_ZOO_HOME/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh` to download the dataset. (It can also be downloaded from Numenta anomaly benchmark [source github](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)).

## Start Jupyter
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-with-jupyter-notebook) to start jupyter notebook. It is recommended to set `SPARK_DRIVER_MEMORY=2g`


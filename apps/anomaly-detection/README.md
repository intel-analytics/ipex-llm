# Anomaly Detection
This is a simple example of unsupervised anomaly detection using Analytics Zoo Keras-Style API. We use RNN to predict following data values based on previous sequence (in order) and measure the distance between predicted values and actual values. If the distance is above some threshold, we report those values as anomaly.

## Environment
* Python 3.5/3.6 (pandas 0.22.0)
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Run Jupyter after pip install
```bash
export SPARK_DRIVER_MEMORY=2g
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

## Run Jupyter with prebuilt package
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package`
* Run `$ANALYTICS_ZOO_HOME/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh` to download the dataset. (It can also be downloaded from its [github](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)).
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[4]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 2g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 2g
```

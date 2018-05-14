# Anomaly Detection
This is a simple example of unsupervised anomaly detection using Analytics Zoo Keras-Style API. We use RNN to predict following data values based on previous sequence (in order) and measure the distance between predicted values and actual values. If the distance is above some threshold, we report those values as anomaly.

## Environment
* Python 2.7/3.5/3.6 (pandas 0.22.0)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics ZOO 0.1.0

## Run with Jupyter
* Download Analytics ZOO and build it.
* Run `export ANALYTICS_ZOO_HOME=the root directory of the Analytics Zoo project`
* Run `$ANALYTICS_ZOO_HOME/data/NAB/nyc_taxi/get_nyc_taxi.sh` to download dataset. (It can also be downloaded from its [github](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv))
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie MASTER = local\[physcial_core_number\]
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/scripts/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 12g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 12g \
```

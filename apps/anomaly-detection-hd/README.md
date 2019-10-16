# Anomaly Detection
This is a simple example of unsupervised anomaly detection using Analytics Zoo Keras-Style API. In statistics, anomalies, also known as outliers, are observation points that are distant from other observations. An outlier may be due to variability in the measurement or it may indicate experimental error, and it can cause serious problems in statistical analyses.
We use an unsupervised neural network to perform a mapping from the original space to another space (encoder) and a mapping back from this new space to the original one (decoder).
Since outliers are rare and different, that the auto-encoder will not learn to map those objects correctly, inducing a higher reconstruction error. This reconstruction error will result in a higher outlier score.
(Retrieved from https://edouardfouche.com/Neural-based-Outlier-Discovery/)

## Environment
* Python 2.7
* Apache Spark 2.2.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Run with Jupyter
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run `$ANALYTICS_ZOO_HOME/bin/data/HiCS/get_HiCS.sh` to download the dataset. (More data can be found from [here](https://www.ipd.kit.edu/~muellere/HiCS/)).
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 32g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 32g
```

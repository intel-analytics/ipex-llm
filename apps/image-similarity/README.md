# image similarity
This is a sample example of image similarity calculation. Both semantic and visually similarity are
introduced. A real estate example was used to recommend similar houses based on the query image
provided by users.

## Environment
* Python 2.7/3.5/3.6
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics Zoo 0.1.0

## Run with Jupyter
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 8g  \
    --total-executor-cores 2  \
    --executor-cores 2  \
    --executor-memory 8g
```
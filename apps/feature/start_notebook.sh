#!/bin/bash

#setup pathes
version=0.4.0
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7/
BigDL_HOME=${HOME}/code/Bigdl/spark-dl
jar=${BigDL_HOME}/spark/dl/target/bigdl-SPARK_2.1-${version}-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=. --ip=* --no-browser"

${SPARK_HOME}/bin/pyspark \
    --master local[2] \
    --driver-cores 1  \
    --driver-memory 10g  \
    --total-executor-cores 3  \
    --executor-cores 1  \
    --executor-memory 20g \
    --py-files ${BigDL_HOME}/spark/dist/target/bigdl-${version}-SNAPSHOT-python-api.zip \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${jar} \
    --conf spark.driver.extraClassPath=${jar} \
    --conf spark.executor.extraClassPath=${jar}

#!/bin/bash

#setup pathes
version=0.4.0
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7/
BigDL=dist-spark-2.1.1-scala-2.11.8-all-0.4.0-dist
jar=${BigDL}/lib/bigdl-SPARK_2.1-${version}-jar-with-dependencies.jar

if [ ! -d $BigDL ]; then
    wget https://s3-ap-southeast-1.amazonaws.com/bigdl-download/$BigDL.zip
    unzip $BigDL.zip -d $BigDL
fi

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
    --py-files ${BigDL}/lib/bigdl-${version}-python-api.zip \
    --properties-file ${BigDL}/conf/spark-bigdl.conf \
    --jars ${jar} \
    --conf spark.driver.extraClassPath=${jar} \
    --conf spark.executor.extraClassPath=${jar}

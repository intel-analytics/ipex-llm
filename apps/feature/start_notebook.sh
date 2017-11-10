#!/bin/bash

#setup pathes
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7/
VISION_HOME=$HOME/code/analytics-zoo/transform/vision/
BigDL_HOME=$HOME/BigDL
MASTER="local[2]"

PYTHON_API_ZIP_PATH=${VISION_HOME}/target/vision-0.1-SNAPSHOT-python-api.zip,${BigDL_HOME}/lib/bigdl-0.3.0-python-api.zip
VISION_JAR_PATH=${BigDL_HOME}/lib/bigdl-SPARK_2.1-0.3.0-jar-with-dependencies.jar,${VISION_HOME}/target/vision-0.1-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=. --ip=* --no-browser"

${SPARK_HOME}/bin/pyspark \
    --master ${MASTER} \
    --driver-cores 1  \
    --driver-memory 10g  \
    --total-executor-cores 3  \
    --executor-cores 1  \
    --executor-memory 20g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
    --jars ${VISION_JAR_PATH} \
    --conf spark.driver.extraClassPath=${VISION_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-SPARK_2.1-0.3.0-jar-with-dependencies.jar,vision-0.1-SNAPSHOT-jar-with-dependencies.jar

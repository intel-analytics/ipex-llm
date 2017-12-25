#!/bin/bash

#setup pathes
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7/
JAR_HOME=$HOME/code/analytics-zoo/models/target
BigDL_HOME=$HOME/code/Bigdl/spark-dl/spark/dist/target/bigdl-0.4.0-SNAPSHOT-spark-2.0.0-scala-2.11.8-all-dist
MASTER="local[2]"

PYTHON_API_ZIP_PATH=$HOME/code/analytics-zoo/models/target/bigdl-models-0.1-SNAPSHOT-python-api.zip
JAR_PATH=${JAR_HOME}/models-0.1-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=* --no-browser"

echo ${PYTHON_API_ZIP_PATH}
echo ${BigDL_HOME}/conf/spark-bigdl.conf
echo ${JAR_PATH}

${SPARK_HOME}/bin/pyspark \
    --master ${MASTER} \
    --driver-cores 1  \
    --driver-memory 10g  \
    --total-executor-cores 3  \
    --executor-cores 1  \
    --executor-memory 20g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
    --jars ${JAR_PATH} \
    --conf spark.driver.extraClassPath=${JAR_PATH} \
    --conf spark.executor.extraClassPath=models-0.1-SNAPSHOT-jar-with-dependencies.jar

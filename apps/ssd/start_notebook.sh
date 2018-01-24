#!/bin/bash

#setup pathes
curr=$PWD
echo $curr
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7
analytics_zoo=$HOME/code/analytics-zoo
MODELS_HOME=${analytics_zoo}/models

PYTHON_API_ZIP_PATH=$MODELS_HOME/target/bigdl-models-0.1-SNAPSHOT-python-api.zip
JAR_PATH=${MODELS_HOME}/target/models-0.1-SNAPSHOT-jar-with-dependencies.jar

# build model zoo

if [ ! -f $PYTHON_API_ZIP_PATH ]
then
  cd $MODELS_HOME
  echo $MODELS_HOME
  bash build.sh
  cd $curr
fi



# when you build models jar, you should have download BigDL
BigDL_HOME=$MODELS_HOME/dist-spark-2.1.1-scala-2.11.8-all-0.4.0-dist
MASTER="local[2]"



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
    --conf spark.driver.maxResultSize=4g \
    --conf spark.executor.extraClassPath=models-0.1-SNAPSHOT-jar-with-dependencies.jar

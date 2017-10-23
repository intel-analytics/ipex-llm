# Demo Setup Guide

## Install Dependency Packages

Reference https://github.com/intel-analytics/BigDL/wiki/Python-Support

## Download BigDL jars

Download BigDL Nightly Build jars from https://github.com/intel-analytics/BigDL/wiki/Downloads

The default spark version is Spark 1.5.1


## 2 Start Jupyter Server

* Create start_notebook.sh, copy and paste the contents below, and edit SPARK_HOME, BigDL_HOME accordingly. Change other parameter settings as you need. 
```bash
#!/bin/bash

#setup pathes
SPARK_HOME=/Users/bigdl/spark-1.6.0-bin-hadoop2.6/
Analytics_HOME=/Users/bigdl/analytics-zoo
BigDL_HOME=/Users/bigdl/dist-spark-1.5.1-scala-2.10.5-linux64-0.2.0-20170510.012057-18-dist
#use local mode or cluster mode
#MASTER=spark://xxxx:7077
MASTER="local[4]"

PYTHON_API_ZIP_PATH=${BigDL_HOME}/lib/bigdl-0.2.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${Analytics_HOME}/pipeline/target/pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export IPYTHON_OPTS="notebook --notebook-dir=./  --ip=* --no-browser --NotebookApp.token=''"

${SPARK_HOME}/bin/pyspark \
    --master ${MASTER} \
    --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
    --driver-cores 1  \
    --driver-memory 10g  \
    --total-executor-cores 3  \
    --executor-cores 1  \
    --executor-memory 20g \
    --conf spark.akka.frameSize=64 \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar
```
* Put start_notebook.sh and start_tensorboard.sh in home directory and execute them in bash.



#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH
export BIGDL_JARS=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

set -e

echo "Start ray horovod pytorch example tests"
#start execute
echo "#1 pytorch estimator example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/horovod/pytorch_estimator.py
now=$(date "+%s")
time1=$((now-start))
echo "horovod pytorch example tests finished"

echo "Start chronos tcmf tests"
#start execute
echo "#2 chronos tcmf example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/tcmf/run_electricity.py --use_dummy_data --smoke
now=$(date "+%s")
time2=$((now-start))
echo "chronos tcmf example tests finished"

echo "#1 pytorch estimator example time used:$time1 seconds"
echo "#2 chronos tcmf example time used:$time2 seconds"

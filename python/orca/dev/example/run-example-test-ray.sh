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

echo "Start ray exmples tests"
#start execute
echo "Start pong example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/rl_pong/rl_pong.py --iterations 10
now=$(date "+%s")
time1=$((now-start))

echo "Start async_parameter example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/parameter_server/async_parameter_server.py --iterations 10
now=$(date "+%s")
time2=$((now-start))

echo "Start sync_parameter example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/parameter_server/sync_parameter_server.py --iterations 10
now=$(date "+%s")
time3=$((now-start))

echo "Start multiagent example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/rllibexample/multiagent_two_trainers.py
now=$(date "+%s")
time4=$((now-start))

echo "End ray example tests"
echo "#9 rl_pong time used:$time1 seconds"
echo "#10 sync_parameter_server time used:$time2 seconds"
echo "#11 async_parameter_server time used:$time3 seconds"
echo "#12 multiagent_two_trainers time used:$time3 seconds"
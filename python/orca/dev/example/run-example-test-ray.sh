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

echo "Start ray example tests"
#start execute
echo "#1 Start rl_pong example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/rl_pong/rl_pong.py --iterations 10
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start async_parameter example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/parameter_server/async_parameter_server.py --iterations 10
now=$(date "+%s")
time2=$((now-start))

echo "#3 Start sync_parameter example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/parameter_server/sync_parameter_server.py --iterations 10
now=$(date "+%s")
time3=$((now-start))

echo "#4 Start multiagent example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/rllib/multiagent_two_trainers.py --iterations 5
now=$(date "+%s")
time4=$((now-start))

echo "#5 Start mxnet lenet example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/mxnet/lenet_mnist.py -e 1 -b 256
now=$(date "+%s")
time5=$((now-start))

echo "Ray example tests finished"
echo "#1 rl_pong time used:$time1 seconds"
echo "#2 async_parameter_server time used:$time2 seconds"
echo "#3 sync_parameter_server time used:$time3 seconds"
echo "#4 multiagent_two_trainers time used:$time4 seconds"
echo "#5 mxnet_lenet time used:$time5 seconds"

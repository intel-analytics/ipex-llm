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

echo "Start ray horovod tf example tests"

echo "#1 tf2 estimator resnet 50 example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf2/resnet/resnet-50-imagenet.py --use_dummy_data --benchmark --batch_size_per_worker 4
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start tf2 estimator lenet"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf2/mnist/lenet_mnist_keras.py --cluster_mode local --max_epoch 1
now=$(date "+%s")
time2=$((now-start))

echo "Ray example tests finished"
echo "#1 tf2 estimator resnet 50 time used:$time1 seconds"
echo "#2 tf2 estimator lenet used:$time2 seconds"

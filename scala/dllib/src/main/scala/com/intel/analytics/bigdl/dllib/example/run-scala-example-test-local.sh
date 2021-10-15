#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "bigdl-dllib*jar-with-dependencies.jar"`
set -e

echo "#1 start example test for LocalEstimator"

if [ -d analytics-zoo-data/data/mnist ]
then
    echo "analytics-zoo-data/data/mnist already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/mnist.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/mnist.zip -d analytics-zoo-data/data/
fi

if [ -d analytics-zoo-data/data/cifar10 ];then
    echo "analytics-zoo-data/data/cifar10 already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/cifar10.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/cifar10.zip -d analytics-zoo-data/data/
fi

echo "##1.1 LenetEstimator testing"

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.bigdl.dllib.examples.localEstimator.LenetLocalEstimator \
-d analytics-zoo-data/data/mnist -b 128 -e 1 -t 4

now=$(date "+%s")
time1=$((now-start))
echo "#1.1 LocalEstimator:LenetEstimator time used:$time2 seconds"

echo "##1.2 ResnetEstimator testing"

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.bigdl.dllib.examples.localEstimator.ResnetLocalEstimator \
-d analytics-zoo-data/data/cifar10 -b 128 -e 1 -t 4

now=$(date "+%s")
time2=$((now-start))

echo "Scala Examples"
echo "#1.1 LocalEstimator:LenetEstimator time used:$time2 seconds"
echo "#1.2 LocalEstimator:ResnetEstimator time used:$time3 seconds"

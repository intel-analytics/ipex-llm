#!/bin/bash

#
# Copyright 2016 The Analytics-Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -x

#setup pathes
ANALYTICS_ZOO_TUTORIALS_HOME=${ANALYTICS_ZOO_HOME}/apps
ANALYTICS_ZOO_VERSION=${ANALYTICS_ZOO_VERSION_ENV}
BIGDL_VERSION=${BIGDL_VERSION_ENV}
SPARK_VERSION=${SPARK_VERSION_ENV}
SPARK_MAJOR_VERSION=${SPARK_VERSION_ENV%%.[0-9]}

export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=$ANALYTICS_ZOO_TUTORIALS_HOME --ip=0.0.0.0 --port=$NotebookPort --no-browser --NotebookApp.token=$NotebookToken --allow-root"

${SPARK_HOME}/bin/pyspark \
  --master local[${RUNTIME_EXECUTOR_CORES_ENV}] \
  --driver-cores ${RUNTIME_DRIVER_CORES_ENV} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY_ENV} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES_ENV} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY_ENV} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES_ENV} \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip \
  --jars ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp

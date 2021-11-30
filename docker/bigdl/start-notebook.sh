#!/bin/bash

#
# Copyright 2016 The BigDL Authors.
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
BIGDL_TUTORIALS_HOME=${BIGDL_HOME}/apps
SPARK_MAJOR_VERSION=${SPARK_VERSION%%.[0-9]}
echo $BIGDL_HOME
echo $BIGDL_VERSION
echo $SPARK_VERSION
echo $SPARK_MAJOR_VERSION

export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=$BIGDL_TUTORIALS_HOME --ip=0.0.0.0 --port=$NOTEBOOK_PORT --no-browser --NotebookApp.token=$NOTEBOOK_TOKEN --allow-root"

echo $RUNTIME_SPARK_MASTER
echo $RUNTIME_EXECUTOR_CORES
echo $RUNTIME_DRIVER_CORES
echo $RUNTIME_DRIVER_MEMORY
echo $RUNTIME_EXECUTOR_CORES
echo $RUNTIME_EXECUTOR_MEMORY
echo $RUNTIME_TOTAL_EXECUTOR_CORES

if [ -z "${KMP_AFFINITY}" ]; then
    export KMP_AFFINITY=granularity=fine,compact,1,0
fi

if [ -z "${OMP_NUM_THREADS}" ]; then
    if [ -z "${ZOO_NUM_MKLTHREADS}" ]; then
        export OMP_NUM_THREADS=1
    else
        if [ `echo $ZOO_NUM_MKLTHREADS | tr '[A-Z]' '[a-z]'` == "all" ]; then
            export OMP_NUM_THREADS=`nproc`
        else
            export OMP_NUM_THREADS=${ZOO_NUM_MKLTHREADS}
        fi
    fi
fi

if [ -z "${KMP_BLOCKTIME}" ]; then
    export KMP_BLOCKTIME=0
fi

# verbose for OpenMP
if [[ $* == *"verbose"* ]]; then
    export KMP_SETTINGS=1
    export KMP_AFFINITY=${KMP_AFFINITY},verbose
fi

${SPARK_HOME}/bin/pyspark \
  --master ${RUNTIME_SPARK_MASTER} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip \
  --jars local://${BIGDL_HOME}/jars/* \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory'

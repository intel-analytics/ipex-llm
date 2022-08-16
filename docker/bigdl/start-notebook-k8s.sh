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

#notebook token and port
port=${port:-12345}
token=${token:-""}

# check the notebook token and port.
while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

#setup pathes
BIGDL_TUTORIALS_HOME=${BIGDL_HOME}/apps
SPARK_MAJOR_VERSION=${SPARK_VERSION%%.[0-9]}
echo $BIGDL_HOME
echo $BIGDL_VERSION
echo $SPARK_VERSION
echo $SPARK_MAJOR_VERSION

export PYSPARK_DRIVER_PYTHON=jupyter-lab
export PYSPARK_DRIVER_PYTHON_OPTS="--notebook-dir=$BIGDL_TUTORIALS_HOME --ip=0.0.0.0 --port=$port --no-browser --NotebookApp.token=$token --allow-root"

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

jars=$(echo ${BIGDL_HOME}/jars/*.jar | tr ' ' ',')
echo $jars

${SPARK_HOME}/bin/pyspark \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode ${DEPLOY_MODE} \
  --conf spark.driver.host=${SPARK_DRIVER_HOST} \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name ${CONTAINER_NAME} \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.kubernetes.executor.deleteOnTermination=false \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=${NFS_CLAIMNAME} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=${NFS_MOUNT_PATH} \
  --conf spark.network.timeout=10000000 \
  --conf spark.executor.heartbeatInterval=10000000 \
  --conf spark.kubernetes.container.image.pullPolicy=Always \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --num-executors ${RUNTIME_EXECUTOR_INSTANCES} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip \
  --conf spark.jars=$jars \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --verbose

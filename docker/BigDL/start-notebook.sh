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

#setup pathes
BIGDL_TUTORIALS_HOME=/opt/work/BigDL-Tutorials
BIGDL_VERSION=${BIGDL_VERSION_ENV}
SPARK_VERSION=${SPARK_VERSION_ENV}
SPARK_MAJOR_VERSION=${SPARK_VERSION_ENV%%.[0-9]}

export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=$BIGDL_TUTORIALS_HOME/notebooks --ip=* --port=$NotebookPort --no-browser --NotebookApp.token=$NotebookToken --allow-root"

#source ${BIGDL_HOME}/bin/bigdl.sh
${SPARK_HOME}/bin/pyspark \
  --master local[4] \
  --driver-memory 4g \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files ${BIGDL_HOME}/lib/bigdl-${BIGDL_VERSION}-python-api.zip \
  --jars ${BIGDL_HOME}/lib/bigdl-SPARK_${SPARK_MAJOR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=${BIGDL_HOME}/lib/bigdl-SPARK_${SPARK_MAJOR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${BIGDL_HOME}/lib/bigdl-SPARK_${SPARK_MAJOR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp

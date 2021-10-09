#!/usr/bin/env bash

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

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
echo "SCRIPT_DIR": $SCRIPT_DIR
export DL_PYTHON_HOME="$(cd ${SCRIPT_DIR}/../src; pwd)"
export BIGDL_HOME="$(cd ${SCRIPT_DIR}/../../..; pwd)"

echo "BIGDL_HOME: $BIGDL_HOME"
echo "SPARK_HOME": $SPARK_HOME
echo "DL_PYTHON_HOME": $DL_PYTHON_HOME

if [ -z ${SPARK_HOME+x} ]; then echo "SPARK_HOME is unset"; exit 1; else echo "SPARK_HOME is set to '$SPARK_HOME'"; fi

export PYSPARK_ZIP=`find $SPARK_HOME/python/lib  -type f -iname '*.zip' | tr "\n" ":"`

export PYTHONPATH=$PYTHONPATH:$PYSPARK_ZIP:$DL_PYTHON_HOME:$BIGDL_HOME/python/dllib/src:$BIGDL_HOME/python/orca/src:$BIGDL_HOME/dist/conf/spark-bigdl.conf
echo "PYTHONPATH": $PYTHONPATH

BIGDL_CLASSPATH=$(find $BIGDL_HOME/dist/lib/ -name "bigdl-friesian-*with-dependencies.jar" | xargs )
BIGDL_CLASSPATH="$(find $BIGDL_HOME/dist/lib/ -name "bigdl-orca-*with-dependencies.jar" | xargs ):$BIGDL_CLASSPATH"
BIGDL_CLASSPATH="$(find $BIGDL_HOME/dist/lib/ -name "bigdl-dllib-*with-dependencies.jar" | xargs ):$BIGDL_CLASSPATH"
export BIGDL_CLASSPATH=$BIGDL_CLASSPATH
echo "BIGDL_CLASSPATH": $BIGDL_CLASSPATH

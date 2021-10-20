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

# This is the default script with maven parameters to release all bigdl packages built on top of
# Spark 2.4.6 for linux.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
BIGDL_DIR="$(cd ${RUN_SCRIPT_DIR}/../..; pwd)"
echo $BIGDL_DIR

if (( $# < 1)); then
  echo "Usage: release_default_linux_spark246.sh version"
  echo "Usage example: bash all_release_default_linux_spark246.sh default"
  echo "Usage example: bash all_release_default_linux_spark246.sh 0.14.0.dev1"
  exit -1
fi

version=$1

DLLIB_SCRIPT_DIR="$(cd ${BIGDL_DIR}/python/dllib/dev/release; pwd)"
echo $DLLIB_SCRIPT_DIR
bash ${DLLIB_SCRIPT_DIR}/release_default_linux_spark246.sh ${version} false

ORCA_SCRIPT_DIR="$(cd ${BIGDL_DIR}/python/orca/dev/release; pwd)"
echo $ORCA_SCRIPT_DIR
bash ${ORCA_SCRIPT_DIR}/all_release_default_linux_spark246.sh ${version} true

FRIESIAN_SCRIPT_DIR="$(cd ${BIGDL_DIR}/python/friesian/dev/release; pwd)"
echo $DLLIB_SCRIPT_DIR
bash ${DLLIB_SCRIPT_DIR}/release_default_linux_spark246.sh ${version} true

CHRONOS_SCRIPT_DIR="$(cd ${BIGDL_DIR}/python/friesian/dev/release; pwd)"
echo $CHRONOS_SCRIPT_DIR
bash ${CHRONOS_SCRIPT_DIR}/release_default_linux_spark246.sh ${version}

SERVING_SCRIPT_DIR="$(cd ${BIGDL_DIR}/python/serving/dev; pwd)"
echo $SERVING_SCRIPT_DIR
bash ${SERVING_SCRIPT_DIR}/release.sh ${version} true

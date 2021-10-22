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

# This is the default script with maven parameters to release bigdl-orca together with
# bigdl-tf and bigdl-math built on top of Spark 3.1.2 for linux.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
BIGDL_DIR="$(cd ${RUN_SCRIPT_DIR}/../../../..; pwd)"
echo $BIGDL_DIR

if (( $# < 3)); then
  echo "Usage: all_release_default_linux_spark312.sh version quick_build upload"
  echo "Usage example: bash all_release_default_linux_spark312.sh default true true"
  echo "Usage example: bash all_release_default_linux_spark312.sh 0.14.0.dev1 false true"
  exit -1
fi

version=$1
quick=$2
upload=$3

bash ${RUN_SCRIPT_DIR}/release_default_linux_spark312.sh ${version} ${quick} ${upload}

TF_SCRIPT_DIR="$(cd ${BIGDL_DIR}/python/tflibs/dev; pwd)"
echo $TF_SCRIPT_DIR
bash ${TF_SCRIPT_DIR}/release_default_linux.sh ${version} ${upload}

MATH_SCRIPT_DIR="$(cd ${BIGDL_DIR}/python/mathlibs/dev; pwd)"
echo $MATH_SCRIPT_DIR
bash ${MATH_SCRIPT_DIR}/release_default_linux.sh ${version} ${upload}

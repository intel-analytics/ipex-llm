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

# This is the default script with maven parameters to release bigdl-orca built on top of
# Spark 2.4.6 for linux.
# Note that if the maven parameters to build bigdl-orca need to be changed,
# make sure to change this file accordingly.
# If you want to customize the release, please use release.sh and specify maven parameters instead.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
ORCA_DIR="$(cd ${RUN_SCRIPT_DIR}/../../; pwd)"
echo $ORCA_DIR
DEV_DIR="$(cd ${ORCA_DIR}/../dev/; pwd)"
echo $DEV_DIR

if (( $# < 4)); then
  echo "Usage: release_default_linux_spark246.sh version quick_build upload suffix"
  echo "Usage example: bash release_default_linux_spark246.sh default true true true"
  echo "Usage example: bash release_default_linux_spark246.sh 0.14.0.dev1 false true false"
  exit -1
fi

version=$1
quick=$2
upload=$3
suffix=$4

if [ ${suffix} == true ]; then
    bash ${DEV_DIR}/add_suffix_spark2.sh $ORCA_DIR/src/setup.py
    bash ${DEV_DIR}/add_suffix_spark2.sh ${RUN_SCRIPT_DIR}/release.sh
else
    bash ${DEV_DIR}/remove_spark_suffix.sh $ORCA_DIR/src/setup.py
    bash ${DEV_DIR}/remove_spark_suffix.sh ${RUN_SCRIPT_DIR}/release.sh
fi

bash ${RUN_SCRIPT_DIR}/release.sh linux ${version} ${quick} ${upload} -Dspark.version=2.4.6 -P spark_2.x

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
# Spark 3.1.2 for mac.
# Note that if the maven parameters to build bigdl-orca need to be changed,
# make sure to change this file accordingly.
# If you want to customize the release, please use release.sh and specify maven parameters instead.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
ORCA_DIR="$(cd ${RUN_SCRIPT_DIR}/../../; pwd)"
echo $ORCA_DIR

if (( $# < 3)); then
  echo "Usage: release_default_mac_spark312.sh version quick_build upload"
  echo "Usage example: bash release_default_mac_spark312.sh default true true"
  echo "Usage example: bash release_default_mac_spark312.sh 0.14.0.dev1 false true"
  exit -1
fi

version=$1
quick=$2
upload=$3

# Add spark3 suffix to the project name to avoid conflict with the whl for spark2.
# Add name=, == and - in pattern matching so that if the script runs twice,
# it won't change anything in the second run.
sed -i "s/bigdl-dllib==/bigdl-dllib-spark3==/g" $ORCA_DIR/src/setup.py
sed -i "s/name='bigdl-orca'/name='bigdl-orca-spark3'/g" $ORCA_DIR/src/setup.py
sed -i "s/dist\/bigdl_orca-/dist\/bigdl_orca_spark3-/g" ${RUN_SCRIPT_DIR}/release.sh

bash ${RUN_SCRIPT_DIR}/release.sh mac ${version} ${quick} ${upload} -Dspark.version=3.1.2 -P spark_3.x

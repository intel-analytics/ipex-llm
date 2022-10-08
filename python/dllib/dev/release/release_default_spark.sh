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

# This is the default script with maven parameters to release bigdl-dllib built on top of Spark for linux or mac.
# Note that if the maven parameters to build bigdl-dllib need to be changed,
# make sure to change this file accordingly.
# If you want to customize the release, please use release.sh and specify maven parameters instead.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
DLLIB_DIR="$(cd ${RUN_SCRIPT_DIR}/../../; pwd)"
echo $DLLIB_DIR
DEV_DIR="$(cd ${DLLIB_DIR}/../dev/; pwd)"
echo $DEV_DIR

if (( $# < 4)); then
  echo "Usage: release_default_spark.sh platform version quick_build upload spark_version suffix"
  echo "Usage example: bash release_default_spark.sh linux default true true 3.1.2 true"
  echo "Usage example: bash release_default_spark.sh mac 0.14.0.dev1 false true 2.4.6 false"
  exit -1
fi

platform=$1
version=$2
quick=$3
upload=$4
spark_version=$5
suffix=$6
profiles=${*:7}

version_array=(${spark_version//./ })
spark_first_version=${version_array[0]}

re='^[2-3]+$'
if ! [[ $spark_first_version =~ $re ]] ; then
   echo "error: Spark version is not a number like 3.1.2"
   exit 1
fi

sed -i "s/pyspark==[0-9].[0-9]*.[0-9]*/pyspark==${spark_version}/g" $DLLIB_DIR/src/setup.py

if [ ${suffix} == true ]; then
    bash ${DEV_DIR}/add_suffix_spark${spark_first_version}.sh $DLLIB_DIR/src/setup.py
    bash ${DEV_DIR}/add_suffix_spark${spark_first_version}.sh ${RUN_SCRIPT_DIR}/release.sh
else
    bash ${DEV_DIR}/remove_spark_suffix.sh $DLLIB_DIR/src/setup.py
    bash ${DEV_DIR}/remove_spark_suffix.sh ${RUN_SCRIPT_DIR}/release.sh
fi

bash ${RUN_SCRIPT_DIR}/release.sh ${platform} ${version} ${quick} ${upload} -Dspark.version=${spark_version} -P spark_${spark_first_version}.x ${profiles} -U

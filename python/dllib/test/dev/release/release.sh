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

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
BIGDL_DIR="$(cd ${RUN_SCRIPT_DIR}/../../../../..; pwd)"
echo $BIGDL_DIR
BIGDL_PYTHON_DIR="$(cd ${BIGDL_DIR}/python/dllib/src; pwd)"
echo $BIGDL_PYTHON_DIR

if (( $# < 2)); then
  echo "Usage: release.sh platform mvn_parameters"
  echo "Usage example: bash release.sh linux false"
  echo "Usage example: bash release.sh linux true"
  echo "you can also add other profiles such as: -Dspark.version=2.4.6 -P spark_2.x"
  exit -1
fi

platform=$1
quick=$2
profiles=${*:3}

bigdl_version=$(cat $BIGDL_DIR/python/version.txt | head -1)

cd ${BIGDL_DIR}/scala
if [ "$platform" ==  "mac" ]; then
    echo "Building bigdl for mac system"
    dist_profile="-P mac $profiles"
    verbose_pname="macosx_10_11_x86_64"
elif [ "$platform" == "linux" ]; then
    echo "Building bigdl for linux system"
    dist_profile="-P linux $profiles"
    verbose_pname="manylinux1_x86_64"
else
    echo "unsupport platform"
fi

bigdl_build_command="bash make-dist.sh ${dist_profile}"
if [ "$quick" == "true" ]; then
    echo "Skip disting BigDL"
else
    echo "Dist BigDL: $bigdl_build_command"
    cd ${BIGDL_DIR}/scala
    $bigdl_build_command
fi

cd $BIGDL_PYTHON_DIR
sdist_command="python setup.py sdist"
echo "packing source code: ${sdist_command}"
$sdist_command

if [ -d "${BIGDL_DIR}/python/dllib/src/build" ]; then
   rm -r ${BIGDL_DIR}/python/dllib/src/build
fi

if [ -d "${BIGDL_DIR}/python/dllib/src/dist" ]; then
   rm -r ${BIGDL_DIR}/python/dllib/src/dist
fi

wheel_command="python setup.py bdist_wheel --plat-name ${verbose_pname}"
echo "Packing python distribution:   $wheel_command"
${wheel_command}

upload_command="twine upload python/dllib/src/dist/BigDL_dllib-${bigdl_version}-py2.py3-none-${verbose_pname}.whl"
echo "Please manually upload with this command:  $upload_command"

#$upload_command

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
BIGDL_DIR="$(cd ${RUN_SCRIPT_DIR}/../../../../; pwd)"
echo $BIGDL_DIR
BIGDL_PYTHON_DIR="$(cd ${RUN_SCRIPT_DIR}/../../../../pyspark; pwd)"
echo $BIGDL_PYTHON_DIR

if (( $# < 2)); then
  echo "Bad parameters. Usage: release.sh mac spark_2.x"
  exit -1
fi

platform=$1
spark_profile=$2
quick=$3
input_version=$4
bigdl_version=$(python -c "exec(open('$BIGDL_DIR/pyspark/bigdl/version.py').read()); print(__version__)")

if [ "$input_version" != "$bigdl_version" ]; then
   echo "Not the proposed version: $bigdl_version"
   exit -1
fi

cd ${BIGDL_DIR}
if [ "$platform" ==  "mac" ]; then
    echo "Building bigdl for mac system"
    dist_profile="-P mac -P $spark_profile"
    verbose_pname="macosx_10_11_x86_64"
elif [ "$platform" == "linux" ]; then
    echo "Building bigdl for linux system"
    dist_profile="-P $spark_profile"
    verbose_pname="manylinux1_x86_64"
else
    echo "unsupport platform"
fi

bigdl_build_command="${BIGDL_DIR}/make-dist.sh ${dist_profile}"
if [ "$quick" == "true" ]; then
    echo "Skip disting BigDL"
else
    echo "Dist BigDL: $bigdl_build_command"
    $bigdl_build_command
fi

cd $BIGDL_PYTHON_DIR
sdist_command="python setup.py sdist"
echo "packing source code: ${sdist_command}"
$sdist_command

if [ -d "${BIGDL_DIR}/pyspark/build" ]; then
   rm -r ${BIGDL_DIR}/pyspark/build
fi

if [ -d "${BIGDL_DIR}/pyspark/dist" ]; then
   rm -r ${BIGDL_DIR}/pyspark/dist
fi
wheel_command="python setup.py bdist_wheel --plat-name ${verbose_pname}"
echo "Packing python distribution:   $wheel_command"
${wheel_command}

upload_command="twine upload dist/BigDL-${bigdl_version}-py2.py3-none-${verbose_pname}.whl"
echo "Please manually upload with this command:  $upload_command"

$upload_command

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
BIGDL_DIR="$(cd ${RUN_SCRIPT_DIR}/../../..; pwd)"
echo $BIGDL_DIR
BIGDL_PYTHON_DIR="$(cd ${BIGDL_DIR}/python/mathlibs/src; pwd)"
echo $BIGDL_PYTHON_DIR


if (( $# < 3)); then
  echo "Usage: release.sh platform version upload"
  echo "Usage example: bash release.sh linux default true"
  echo "Usage example: bash release.sh mac 0.1.0.dev0 false"
  exit -1
fi

platform=$1
version=$2
upload=$3  # Whether to upload the whl to pypi

if [ "${version}" != "default" ]; then
    echo "User specified version: ${version}"
    sed -i "s/\(__version__ =\)\(.*\)/\1 \"${version}\"/" $BIGDL_PYTHON_DIR/__init__.py
fi

effect_version=`cat $BIGDL_PYTHON_DIR/__init__.py | grep "__version__" | awk '{print $NF}' | tr -d '"'`
echo "The effective version is: ${effect_version}"

if [ "$platform" ==  "mac" ]; then
    verbose_pname="macosx_10_11_x86_64"
elif [ "$platform" == "linux" ]; then
    verbose_pname="manylinux2010_x86_64"
else
    echo "Unsupported platform"
fi

if [ -d "${BIGDL_DIR}/python/mathlibs/src/build" ]; then
   rm -r ${BIGDL_DIR}/python/mathlibs/src/build
fi

if [ -d "${BIGDL_DIR}/python/mathlibs/src/dist" ]; then
   rm -r ${BIGDL_DIR}/python/mathlibs/src/dist
fi

if [ -d "${BIGDL_DIR}/python/mathlibs/src/bigdl_math.egg-info" ]; then
   rm -r ${BIGDL_DIR}/python/mathlibs/src/bigdl_math.egg-info
fi

cd $BIGDL_PYTHON_DIR
wheel_command="python setup.py bdist_wheel --plat-name ${verbose_pname} --python-tag py3"
echo "Packing python distribution: $wheel_command"
${wheel_command}

if [ ${upload} == true ]; then
    upload_command="twine upload dist/bigdl_math-${effect_version}-py3-none-${verbose_pname}.whl"
    echo "Please manually upload with this command: $upload_command"
    $upload_command
fi

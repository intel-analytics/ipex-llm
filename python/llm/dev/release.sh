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
BIGDL_PYTHON_DIR="$(cd ${BIGDL_DIR}/python/llm; pwd)"
echo $BIGDL_PYTHON_DIR

if (( $# < 3)); then
  echo "Usage: release.sh platform version upload"
  echo "Usage example: bash release.sh linux default true"
  exit -1
fi

platform=$1
version=$2
upload=$3  # Whether to upload the whl to pypi

if [ "${version}" != "default" ]; then
    echo "User specified version: ${version}"
    echo $version > $BIGDL_DIR/python/llm/version.txt
fi

ipex_llm_version=$(cat $BIGDL_DIR/python/llm/version.txt | head -1)
echo "The effective version is: ${ipex_llm_version}"

if [ "$platform" == "linux" ]; then
    verbose_pname="manylinux2010_x86_64"
    platform_name="--linux"
elif [ "$platform" == "windows" ]; then
    verbose_pname="win_amd64"
    platform_name="--win"
else
    echo "Unsupported platform"
fi

if [ -d "${BIGDL_DIR}/python/llm/dist" ]; then
   rm -r ${BIGDL_DIR}/python/llm/dist
fi

if [ -d "${BIGDL_DIR}/python/llm/build" ]; then
   rm -r ${BIGDL_DIR}/python/llm/build
fi

cd $BIGDL_PYTHON_DIR

wheel_command="python setup.py clean --all bdist_wheel ${platform_name} --plat-name ${verbose_pname} --python-tag py3"

echo "Packing python distribution: $wheel_command"
${wheel_command}

if [ -d "${BIGDL_DIR}/python/llm/build" ]; then
   rm -r ${BIGDL_DIR}/python/llm/build
fi

if [ ${upload} == true ]; then
    # upload to pypi
    upload_to_pypi_command="twine upload dist/ipex_llm-${ipex_llm_version}-*-${verbose_pname}.whl"
    echo "Please manually upload with this command: $upload_to_pypi_command"
    $upload_to_pypi_command
fi

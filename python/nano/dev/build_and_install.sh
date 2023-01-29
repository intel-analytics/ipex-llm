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
BIGDL_DIR="$(cd ${RUN_SCRIPT_DIR}/../../..; pwd)"
WHL_DIR="$(cd ${BIGDL_DIR}/python/nano; pwd)"

if (( $# < 4)); then
  echo "Usage: build_and_install.sh platform version upload framework pip_install_options[optional]"
  echo "Usage example: bash build_and_install.sh linux default true pytorch --force-reinstall"
  echo "Usage example: bash build_and_install.sh mac 0.14.0.dev1 false tensorflow"
  echo "To install Nano without any framework, specify framework to basic"
  echo "Usage example: bash build_and_install.sh linux default true basic"
  exit -1
fi

platform=$1
version=$2
upload=$3  # Whether to upload the whl to pypi
framework=$4
install_options=${@:5:$((${#@}))}

bash ${RUN_SCRIPT_DIR}/release.sh ${platform} ${version} ${upload}

cd ${WHL_DIR}
whl_name=`ls dist`
if [ $framework == "basic" ];
then
  # when framework is specified to "basic", nano will be installed without
  # any framework like Pytorch and TensorFlow
  pip install $install_options dist/${whl_name}
else
  pip install $install_options dist/${whl_name}[${framework}]
fi

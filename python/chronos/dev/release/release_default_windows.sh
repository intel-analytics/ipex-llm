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

# This is the default script  parameters to release bigdl-chronos for windows.
# Note that if the parameters to build bigdl-chronos need to be changed,
# make sure to change this file accordingly.
# If you want to customize the release, please use release.sh and specify parameters instead.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR

if (( $# < 2)); then
  echo "Usage: release_default_windows.sh version upload"
  echo "Usage example: bash release_default_windows.sh default true"
  echo "Usage example: bash release_default_windows.sh 0.14.0.dev1 true"
  exit -1
fi

version=$1
upload=$2

bash ${RUN_SCRIPT_DIR}/release.sh windows ${version} ${upload}

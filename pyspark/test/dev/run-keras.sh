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

# run: run-keras.sh python2.7
. `dirname $0`/prepare_env.sh

if (( $# < 1)); then
  echo "Bad parameters. Usage: run-keras.sh python2.7"
  exit -1
fi

cd "`dirname $0`"
p=$1  # executable python 
export DL_CORE_NUMBER=4

echo "${cyan}Using python version: $p${reset}"
export PYTHON_EXECUTABLE=$p
export PYSPARK_PYTHON=$p
export PYSPARK_DRIVER_PYTHON=$p
$p -m pytest -v  ../../../pyspark/test/bigdl/keras

exit_status=$?
if [ $exit_status -ne 0 ];
then
    exit $exit_status
fi


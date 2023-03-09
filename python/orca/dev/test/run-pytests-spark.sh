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

# bigdl orca test only support pip, you have to install orca whl before running the script.
#. `dirname $0`/prepare_env.sh

set -ex

cd "`dirname $0`"
cd ../..

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

python_version=$(python --version | awk '{print$2}')
if [ $python_version == 3.7.10 ];then
  # test_estimator_openvino.py is tested in run-pytests-spark-openvino.sh
  echo "Running orca tfpark tests"
  python -m pytest -v test/bigdl/orca/tfpark
  exit_status_1=$?
  if [ $exit_status_1 -ne 0 ];
  then
      exit $exit_status_1
  fi
  python -m pytest -v test/bigdl/orca/learn/bigdl --ignore=test/bigdl/orca/learn/bigdl/test_estimator_openvino.py
  exit_status_2=$?
  if [ $exit_status_2 -ne 0 ];
  then
      exit $exit_status_2
  fi
fi

echo "Running orca data tests"
# test_xshards_partition.py is tested in run-pytests-basic-env.sh
# ray related tests are tested in run-pytests-ray.sh
python -m pytest -v test/bigdl/orca/data \
      --ignore=test/bigdl/orca/data/test_xshards_partition.py \
      --ignore=test/bigdl/orca/data/ray
exit_status_3=$?
if [ $exit_status_3 -ne 0 ];
then
   exit $exit_status_3
fi

echo "Running orca learn metrics tests"
python -m pytest -v test/bigdl/orca/learn/test_metrics.py
exit_status_4=$?
if [ $exit_status_4 -ne 0 ];
then
   exit $exit_status_4
fi

echo "Running orca learn utils tests"
python -m pytest -v test/bigdl/orca/learn/test_utils.py
exit_status_5=$?
if [ $exit_status_5 -ne 0 ];
then
   exit $exit_status_5
fi

echo "Running orca learn inference tests"
python -m pytest -v test/bigdl/orca/inference
exit_status_6=$?
if [ $exit_status_6 -ne 0 ];
then
   exit $exit_status_6
fi

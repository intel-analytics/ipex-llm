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

cd "`dirname $0`"
cd ../..

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

ray stop -f

set -ex

echo "Running RayOnSpark tests"
python -m pytest -v test/bigdl/orca/ray/ \
    --ignore=test/bigdl/orca/ray/integration/ \
    --ignore=test/bigdl/orca/ray/ray_cluster/
exit_status_1=$?
if [ $exit_status_1 -ne 0 ];
then
    exit $exit_status_1
fi
ray stop -f

# TODO: fix extra fixture {"spark.python.worker.reuse": "false"}
#       for tensorflow estimator ray backend unit tests
python -m pytest -v test/bigdl/orca/learn/ray/tf/
exit_status_2=$?
if [ $exit_status_2 -ne 0 ];
then
    exit $exit_status_2
fi
ray stop -f

echo "Running orca learn ray backend tests"
python -m pytest -v test/bigdl/orca/learn/ray \
      --ignore=test/bigdl/orca/learn/ray/horovod/ \
      --ignore=test/bigdl/orca/learn/ray/ctx/ \
      --ignore=test/bigdl/orca/learn/ray/tf/ \
      --ignore=test/bigdl/orca/learn/ray/mxnet/
exit_status_3=$?
if [ $exit_status_3 -ne 0 ];
then
    exit $exit_status_3
fi
ray stop -f

echo "Running orca data ray related tests"
python -m pytest -v test/bigdl/orca/data/ray
exit_status_4=$?
if [ $exit_status_4 -ne 0 ];
then
    exit $exit_status_4
fi
ray stop -f

echo "Running orca learn spark backend tests"
python -m pytest -v test/bigdl/orca/learn/spark/
exit_status_6=$?
if [ $exit_status_6 -ne 0 ];
then
    exit $exit_status_6
fi

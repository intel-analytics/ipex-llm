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

ray stop -f
ray start --head

echo "Running Ray Estimator tests"
python test/bigdl/orca/learn/ray/pytorch/test_estimator_ray_runtime.py
exit_status_2=$?
if [ $exit_status_2 -ne 0 ];
then
    exit $exit_status_2
fi

python test/bigdl/orca/learn/ray/tf/test_tf2estimator_ray_runtime.py
exit_status_3=$?
if [ $exit_status_3 -ne 0 ];
then
    exit $exit_status_3
fi

ray stop -f
ray start --head

echo "Running RayContext tests"
python -m pytest -v test/bigdl/orca/ray/ray_cluster
exit_status_1=$?
if [ $exit_status_1 -ne 0 ];
then
    exit $exit_status_1
fi

echo "Running PyTorch Estimator Ray Dataset tests"
python -m pytest -v test/bigdl/orca/learn/ray/pytorch/test_estimator_ray_dataset.py
exit_status_4=$?
if [ $exit_status_4 -ne 0 ];
then
    exit $exit_status_4
fi

echo "Running TF2Estimator Ray Dataset tests"
python -m pytest -v test/bigdl/orca/learn/ray/tf/test_tf2estimator_ray_dataset.py
exit_status_5=$?
if [ $exit_status_5 -ne 0 ];
then
    exit $exit_status_5
fi

ray stop -f

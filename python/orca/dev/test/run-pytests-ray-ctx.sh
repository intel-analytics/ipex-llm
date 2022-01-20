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

echo "Running RayContext tests"
python -m pytest -v test/bigdl/orca/ray/ray_cluster/ray_ctx
exit_status_1=$?
if [ $exit_status_1 -ne 0 ];
then
    exit $exit_status_1
fi

echo "Running Ray Estimator tests"
python ${BIGDL_ROOT}/python/orca/test/bigdl/orca/ray/ray_cluster/ray_est/test_ray_tf2estimator.py
python ${BIGDL_ROOT}/python/orca/test/bigdl/orca/ray/ray_cluster/ray_est/test_ray_pytorch_estimator.py
exit_status_1=$?
if [ $exit_status_1 -ne 0 ];
then
    exit $exit_status_1
fi

ray stop -f

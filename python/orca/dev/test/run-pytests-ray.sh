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

echo "Running RayOnSpark tests"
python -m pytest -v test/bigdl/orca/ray/ \
    --ignore=test/bigdl/orca/ray/integration/ \
    --ignore=test/bigdl/orca/ray/ray_cluster/ \
    --ignore=test/bigdl/orca/ray/test_reinit_raycontext.py
exit_status_1=$?
if [ $exit_status_1 -ne 0 ];
then
    exit $exit_status_1
fi

ray stop -f

echo "Running orca learn ray backend tests"
python -m pytest -v test/bigdl/orca/learn/ray \
      --ignore=test/bigdl/orca/learn/ray/pytorch/test_estimator_horovod_backend.py \
      --ignore=test/bigdl/orca/learn/ray/pytorch/test_estimator_ray_runtime.py \
      --ignore=test/bigdl/orca/learn/ray/pytorch/test_estimator_ray_dataset.py \
      --ignore=test/bigdl/orca/learn/ray/tf/ \
      --ignore=test/bigdl/orca/learn/ray/mxnet/
exit_status_2=$?
if [ $exit_status_2 -ne 0 ];
then
   exit $exit_status_2
fi

echo "Running orca learn tf2 ray backend tests"
python -m pytest -v test/bigdl/orca/learn/ray/tf/test_tf2estimator_ray_backend.py
exit_status_3=$?
if [ $exit_status_3 -ne 0 ];
then
   exit $exit_status_3
fi

echo "Running orca data tests"
# test_xshards_partition.py is tested in run-pytests-basic-env.sh
# test_write_parquet.py is tested in run-pytests-spark.sh
python -m pytest -v test/bigdl/orca/data \
      --ignore=test/bigdl/orca/data/test_xshards_partition.py \
      --ignore=test/bigdl/orca/data/test_write_parquet.py
exit_status_4=$?
if [ $exit_status_4 -ne 0 ];
then
    exit $exit_status_4
fi

python_version=$(python --version | awk '{print$2}')
if [ $python_version == 3.7.10 ];then
    echo "Running orca mxnet tests"
    python -m pytest -v test/bigdl/orca/learn/ray/mxnet/
    exit_status_5=$?
    if [ $exit_status_5 -ne 0 ];then
        exit $exit_status_5
    fi
fi
ray stop -f

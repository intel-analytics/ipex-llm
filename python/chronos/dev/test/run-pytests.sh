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


cd "`dirname $0`"
cd ../..

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python
if [ -z "${OMP_NUM_THREADS}" ]; then
    export OMP_NUM_THREADS=1
fi

ray stop -f

echo "Running chronos tests"
python -m pytest -v test/bigdl/chronos \
       -k "not test_forecast_tcmf_distributed"
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi

ray stop -f


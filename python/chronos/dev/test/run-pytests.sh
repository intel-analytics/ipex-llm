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

RUN_PART1=0
RUN_PART2=0
if [ $1 = 1 ]; then
RUN_PART1=1
RUN_PART2=0
elif [ $1 = 2 ]; then
RUN_PART1=0
RUN_PART2=1
else
RUN_PART1=1
RUN_PART2=1
fi

if [ $RUN_PART1 = 1 ]; then
echo "Running chronos tests Part 1"
python -m pytest -v test/bigdl/chronos/model
python -m pytest -v test/bigdl/chronos/forecaster\
       -k "not test_forecast_tcmf_distributed"
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi
fi

if [ $RUN_PART2 = 1 ]; then
echo "Running chronos tests Part 2"
python -m pytest -v test/bigdl/chronos/autots\
                    test/bigdl/chronos/data \
                    test/bigdl/chronos/simulator \
                    test/bigdl/chronos/detector \
       -k "not test_forecast_tcmf_distributed"
exit_status_0=$?
if [ $exit_status_0 -ne 0 ];
then
    exit $exit_status_0
fi
fi

ray stop -f


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


set -ex

export FTP_URI=$FTP_URI
export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

ray stop -f

cd "`dirname $0`"
cd ../../tutorial/xshards

echo "Running Xshards tests"

#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/train.csv ]
then
    echo "train.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/xshards/train.csv -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

python tabular_playground_series.py

now=$(date "+%s")
time1=$((now - start))

echo "Running Xshards tests time used: $time1 seconds"

echo "Running Xshards tests 2"

#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/titanic.csv ]
then
    echo "titanic.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/xshards/titanic.csv -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

python titanic.py

rm -rf result

now=$(date "+%s")
time2=$((now - start))

echo "Running Xshards tests 1 time used: $time2 seconds"

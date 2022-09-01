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

echo "#1 Running run-tabular_playground_series"
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

echo "#2 Running titanic"
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

echo "#3 Running diabetes"
#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/pima-indians-diabetes.csv ]
then
    echo "pima-indians-diabetes.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/xshards/pima-indians-diabetes.csv -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

sed -i "s/epochs=150/epochs=2/g" diabetes.py
python diabetes.py

now=$(date "+%s")
time3=$((now - start))


echo "#4 Running ionosphere"
#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/ionosphere.csv ]
then
    echo "ionosphere.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/xshards/ionosphere.csv -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

sed -i "s/epochs=100/epochs=2/g" ionosphere.py
python ionosphere.py

now=$(date "+%s")
time4=$((now - start))


echo "#5 Running auto_mpg"
#timer
start=$(date "+%s")

if [ -f ${BIGDL_ROOT}/python/orca/tutorial/xshards/auto-mpg.csv ]
then
    echo "auto-mpg.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/xshards/auto-mpg.csv -P ${BIGDL_ROOT}/python/orca/tutorial/xshards/
fi

sed -i "s/EPOCHS = 1000/EPOCHS = 2/g" auto_mpg.py
python auto_mpg.py

now=$(date "+%s")
time5=$((now - start))

echo "#1 Running run-tabular_playground_series time used: $time1 seconds"
echo "#2 Running titanic time used: $time2 seconds"
echo "#3 Running diabetes time used: $time3 seconds"
echo "#4 Running ionosphere time used: $time4 seconds"
echo "#5 Running auto_mpg time used: $time5 seconds"

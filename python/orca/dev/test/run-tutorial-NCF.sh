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

# clean train/predict/resume results
function clean () {
    echo "Cleaning files..."
    rm -rf NCF_model
    rm -f NCF_model
    rm -rf NCF_resume_model
    rm -f NCF_resume_model
    rm -f *.h5
    rm -rf ./test*
    rm -rf ./train*
    rm -f config.json
    rm -rf logs/
    echo "done"
}

function stop_ray () {
    if [ $1 = "ray" ]; then
        echo "Trying to stop any activate ray context..."
        ray stop -f
    else 
        echo "Backend is not ray, skipping"
    fi
}

set -ex

export FTP_URI=$FTP_URI
export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

cd "`dirname $0`"
cd ../../tutorial/NCF

# 1st argument represents the backend, either "ray" or "spark"
# 2nd argument represents the dataset, either "ml-1m" or "ml-100k"
# if no argument is provided, default to be "spark", "ml-1m"
argc=$#
if [ $argc -eq 0 ]; then
    backend="spark"
    dataset="ml-1m"
else
    backend=$1
    dataset=$2
fi
echo "Start Orca NCF tutorial Test - $backend backend, $dataset dataset"

# download dataset from ftp
rm -rf ml-1m
rm -rf ml-100k
if [ -f ./orca-tutorial-ncf-dataset-compressed.zip ]
then
    echo "dataset ml-1m already exists"
else
    wget $FTP_URI/analytics-zoo-data/orca-tutorial-ncf-dataset-compressed.zip
fi
if [ -f ./orca-tutorial-ncf-dataset-compressed-100k.zip ]
then
    echo "dataset ml-100k already exists"
else
    wget $FTP_URI/analytics-zoo-data/orca-tutorial-ncf-dataset-compressed-100k.zip
fi
unzip orca-tutorial-ncf-dataset-compressed.zip
unzip orca-tutorial-ncf-dataset-compressed-100k.zip
echo "Successfully got dataset ml-1m & ml-100k from ftp"

stop_ray $backend

echo "#1 Running pytorch dataloader"
#timer
start=$(date "+%s")

python ./pytorch_train_dataloader.py --backend $backend --dataset $dataset
# pytorch dataloader does not have predict
python ./pytorch_resume_train_dataloader.py --backend $backend --dataset $dataset

now=$(date "+%s")
time1=$((now - start))

clean
stop_ray $backend

echo "#2 Running pytorch spark dataframe"
#timer
start=$(date "+%s")

python ./pytorch_train_spark_dataframe.py --backend $backend --dataset $dataset
python ./pytorch_predict_spark_dataframe.py --backend $backend --dataset $dataset
python ./pytorch_resume_train_spark_dataframe.py --backend $backend --dataset $dataset

now=$(date "+%s")
time2=$((now - start))

clean
stop_ray $backend

echo "#3 Running pytorch xshards"
#timer
start=$(date "+%s")

python ./pytorch_train_xshards.py --backend $backend --dataset $dataset
python ./pytorch_predict_xshards.py --backend $backend --dataset $dataset
python ./pytorch_resume_train_xshards.py --backend $backend --dataset $dataset

now=$(date "+%s")
time3=$((now - start))

clean
stop_ray $backend

echo "#4 Running tensorflow spark dataframe"
#timer
start=$(date "+%s")

python ./tf_train_spark_dataframe.py --backend $backend --dataset $dataset
python ./tf_predict_spark_dataframe.py --backend $backend --dataset $dataset
python ./tf_resume_train_spark_dataframe.py --backend $backend --dataset $dataset

now=$(date "+%s")
time4=$((now - start))

clean
stop_ray $backend

echo "#5 Running tensorflow xshards"
#timer
start=$(date "+%s")

python ./tf_train_xshards.py --backend $backend --dataset $dataset
python ./tf_predict_xshards.py --backend $backend --dataset $dataset
python ./tf_resume_train_xshards.py --backend $backend --dataset $dataset

now=$(date "+%s")
time5=$((now - start))

clean
stop_ray $backend

echo "#1 Running pytorch dataloader time used: $time1 seconds"
echo "#2 Running pytorch spark dataframe time used: $time2 seconds"
echo "#3 Running pytorch xshards time used: $time3 seconds"
echo "#4 Running tensorflow spark dataframe time used: $time4 seconds"
echo "#5 Running tensorflow xshards time used: $time5 seconds"

#clean dataset
rm -rf ml-1m
rm -rf ml-100k

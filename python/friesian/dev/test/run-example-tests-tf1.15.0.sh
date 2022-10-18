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

export FTP_URI=$FTP_URI
set -e
mkdir -p result

echo "#1 start example test for dien train"
#timer
start=$(date "+%s")
if [ -d data/dien_preprocessed ]; then
  echo "data/dien_preprocessed already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/dien_preprocessed.tar.gz -P data
  tar -xvzf data/dien_preprocessed.tar.gz -C data
fi

python ../../example/dien/dien_train.py \
    --executor_cores 4 \
    --executor_memory 50g \
    --batch_size 128 \
    --data_dir ./data/dien_preprocessed \
    --model_dir ./result

now=$(date "+%s")
time1=$((now - start))

echo "#2 start example pytorch test for dien train"
#timer
start=$(date "+%s")

python ../../example/dien/pytorch/dien_train.py \
    --executor_cores 4 \
    --executor_memory 50g \
    --num_executors 2 \
    --epochs 10 \
    --batch_size 128 \
    --data_dir ./data/dien_preprocessed/

now=$(date "+%s")
time2=$((now - start))

#echo "#3 start example deeprec wdl train"
##timer
#start=$(date "+%s")
#if [ -d data/input_deeprec ]; then
#  echo "data/input_deeprec already exists"
#else
#  wget -nv $FTP_URI/analytics-zoo-data/input_deeprec.tar.gz -P data
#  tar -xvzf data/input_deeprec.tar.gz -C data
#fi
#
#python ../../example/deeprec/wdl.py \
#    --smartstaged false \
#    --data_location ./data/input_deeprec \
#    --checkpoint ./result \
#    --instances_per_node 3
#
#now=$(date "+%s")
#time3=$((now - start))

rm -rf data
rm -rf result

echo "#1 dien train time used: $time1 seconds"
echo "#2 dien train with pytorch time used: $time2 seconds"
#echo "#3 deeprec wdl time used: $time3 seconds"

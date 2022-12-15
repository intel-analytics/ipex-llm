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
mkdir -p result/stats

echo "#1 start example test for two tower train"
#timer
start=$(date "+%s")
if [ -d data/input_2tower ]; then
  echo "data/input_2tower already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/input_2tower.tar.gz -P data
  tar -xvzf data/input_2tower.tar.gz -C data
fi

python ../../example/two_tower/train_2tower.py \
    --executor_cores 4 \
    --executor_memory 10g \
    --data_dir ./data/input_2tower \
    --model_dir ./result \
    --frequency_limit 2

now=$(date "+%s")
time1=$((now - start))

if [ -d data/recsys_sample ]; then
  echo "data/recsys_sample already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/recsys_sample.tar.gz -P data
  tar -xvzf data/recsys_sample.tar.gz -C data
fi

echo "#2 start example test for wnd recsys2021 train"
#timer
start=$(date "+%s")
python ../../example/wnd/recsys2021/wnd_train_recsys.py \
    --executor_cores 4 \
    --executor_memory 10g \
    --data_dir ./data/recsys_sample \
    --model_dir ./result \
    -b 1600

now=$(date "+%s")
time2=$((now - start))

echo "#3 start example test for xgboost train"
#timer
start=$(date "+%s")
sed -i "s/\[0.1, 0.15, 0.2, 0.25, 0.3\]/\[0.1\]/g" ../../example/xgb/xgb_train.py
sed -i "s/\[6, 8, 10, 12\]/\[6\]/g" ../../example/xgb/xgb_train.py
sed -i "s/\[200, 400, 600, 800, 1000\]/\[200\]/g" ../../example/xgb/xgb_train.py
python ../../example/xgb/xgb_train.py \
    --executor_cores 4 \
    --executor_memory 10g \
    --data_dir ./data/recsys_sample \
    --model_dir ./result/xgb_res

now=$(date "+%s")
time3=$((now - start))

echo "#4 start example test for ncf train"
start=$(date "+%s")
if [ -d data/movielens ]; then
  echo "data/movielens already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/movielens.tar.gz -P data
  tar -xvzf data/movielens.tar.gz -C data
fi

python ../../example/ncf/ncf_train.py\
    --data_dir ./data/movielens \
    --model_dir ./result/ \
    --epochs 1

python ../../example/ncf/ncf_train.py\
    --data_dir ./data/movielens \
    --model_dir ./result/ \
    --epochs 1 \
    --backend spark

now=$(date "+%s")
time4=$((now - start))

echo "#5 start example test for deepfm train"
start=$(date "+%s")

python ../../example/deep_fm/deepFM_train.py \
    --data_dir ./data/recsys_sample \
    --model_dir ./result/deepFM_model \
    --frequency_limit 1

now=$(date "+%s")
time5=$((now - start))

echo "#6 start example test for multi task train"

if [ -d data/multi_task_data.csv ]; then
  echo "data/multi_task_data.csv already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/multi_task_data.csv -P data
fi

python ../../example/multi_task/data_processing.py \
    --input_path ./data/multi_task_data.csv \
    --output_path ./data


start=$(date "+%s")

python ../../example/multi_task/run_multi_task.py \
    --model_save_path ./result/multi_task \
    --train_data_path ./data/train_multi_task \
    --test_data_path ./data/test_multi_task \

now=$(date "+%s")
time6=$((now - start))

echo "#7 start example test for wnd preprocessing and train"
#timer
start=$(date "+%s")
if [ -f data/dac_sample.txt ]; then
  echo "data/dac_sample.txt already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/dac_sample.tar.gz -P data
  tar -xvzf data/dac_sample.tar.gz -C data
fi
python ../../example/wnd/csv_to_parquet.py \
        --input ./data/dac_sample.txt \
        --output ./data/day_0.parquet
cp -r ./data/day_0.parquet ./data/day_1.parquet
python ../../example/wnd/wnd_preprocessing.py \
    --executor_cores 6 \
    --executor_memory 50g \
    --days 0-1 \
    --input_folder ./data \
    --output_folder ./result \
    --frequency_limit 15 \
    --cross_sizes 10000,10000

python ../../example/wnd/wnd_train.py \
    --executor_cores 6 \
    --executor_memory 50g \
    --data_dir ./result \
    --model_dir ./wnd_model
now=$(date "+%s")
time7=$((now - start))

rm -rf data
rm -rf result

echo "#1 two tower train time used: $time1 seconds"
echo "#2 wnd recsys2021 train time used: $time2 seconds"
echo "#3 xgboost train time used: $time3 seconds"
echo "#4 ncf train time used: $time4 seconds"
echo "#5 deepfm train time used: $time5 seconds"
echo "#6 multi_task train time used: $time6 seconds"
echo "#7 wnd criteo preprocessing and train time used: $time7 seconds"

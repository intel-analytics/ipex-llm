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
echo "#1 start example test for dien preprocessing"
#timer
start=$(date "+%s")
if [ -f data/test.json ]; then
  echo "data/test.json already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/test.json -P data
fi
if [ -f data/test ]; then
  echo "data/test already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/test -P data
fi

python ../../example/dien/dien_preprocessing.py \
    --cluster_mode standalone \
    --executor_cores 4 \
    --num_executors 2 \
    --executor_memory 2g \
    --driver_cores 2 \
    --driver_memory 1g \
    --input_meta ./data/test \
    --input_transaction ./data/test.json \
    --output ./result/

now=$(date "+%s")
time1=$((now - start))

echo "#2 start example test for preprocessing inference"
#timer
start=$(date "+%s")
if [ -f data/amazon_books_vocs.tar.gz ]; then
  echo "data/amazon_books_vocs.tar.gz already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/amazon_books_vocs.tar.gz -P data
  tar -xf data/amazon_books_vocs.tar.gz -C data
fi
if [ -f data/reviews_Books.json ]; then
  echo "data/reviews_Books.json already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/reviews_Books.json -P data
fi
if [ -f data/meta_Books.json ]; then
  echo "data/meta_Books.json already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/meta_Books.json -P data
fi

python ../../example/dien/preprocessing_inference.py \
    --cluster_mode standalone \
    --executor_cores 4 \
    --num_executors 2 \
    --executor_memory 10g \
    --driver_cores 2 \
    --driver_memory 1g \
    --input_transaction ./data/reviews_Books.json \
    --input_meta ./data/meta_Books.json \
    --index_folder ./data/ --num_save_files 80

now=$(date "+%s")
time2=$((now - start))

rm -rf data
rm -rf result

echo "#1 dien preprocessing time used: $time1 seconds"
echo "#2 preprocessing inference time used: $time2 seconds"

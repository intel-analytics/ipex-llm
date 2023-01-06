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
cd ../../tutorial/NCF

#download dataset from ftp
rm -f ./orca-tutorial-ncf-dataset.zip
rm -rf ml-1m
wget $FTP_URI/analytics-zoo-data/orca-tutorial-ncf-dataset.zip
unzip orca-tutorial-ncf-dataset.zip
echo "Successfully gets dataset from ftp"

echo "#1 Running pytorch_train_dataloader"
#timer
start=$(date "+%s")

python ./pytorch_train_dataloader.py

now=$(date "+%s")
time1=$((now - start))

#clean files
rm -rf runs
rm -f NCF_model

echo "#2 Running pytorch_train_spark_dataframe"
#timer
start=$(date "+%s")

python ./pytorch_train_spark_dataframe.py

now=$(date "+%s")
time2=$((now - start))

#clean files
rm -rf runs
rm -f NCF_model

echo "#3 Running pytorch_train_xshards"
#timer
start=$(date "+%s")

python ./pytorch_train_xshards.py

now=$(date "+%s")
time3=$((now - start))

#clean files
rm -rf runs
rm -f NCF_model

echo "#4 Running tf_train_spark_dataframe"
#timer
start=$(date "+%s")

python ./tf_train_spark_dataframe.py

now=$(date "+%s")
time4=$((now - start))

#clean files
rm -rf test_processed.parquet
rm -rf train_processed.parquet
rm -rf NCF_model
rm -rf log

echo "#5 Running tf_train_xshards"
#timer
start=$(date "+%s")

python ./tf_train_xshards.py

now=$(date "+%s")
time5=$((now - start))

#clean files
rm -rf test_processed.parquet
rm -rf train_processed.parquet
rm -rf NCF_model
rm -rf log

echo "#1 Running pytorch_train_dataloader time used: $time1 seconds"
echo "#2 Running pytorch_train_spark_dataframe time used: $time2 seconds"
echo "#3 Running pytorch_train_xshards time used: $time3 seconds"
echo "#4 Running tf_train_spark_dataframe time used: $time4 seconds"
echo "#5 Running tf_train_xshards time used: $time5 seconds"

#clean files
rm -r ml-1m
rm orca-tutorial-ncf-dataset.zip
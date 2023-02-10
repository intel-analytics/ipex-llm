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
function clean() {
    echo "Cleaning files..."
    rm -rf NCF_model
    rm -f NCF_model
    rm -rf NCF_resume_model
    rm -f NCF_resume_model
    rm -rf ./ml-1m/test*
    rm -rf ./ml-1m/train*
    rm -f config.json
    echo "done"
}

set -ex

export FTP_URI=$FTP_URI
export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

ray stop -f

cd "`dirname $0`"
cd ../../tutorial/NCF

# backend passed as the first argument, either "ray" or "spark"
# if no argument is provided, default to be "spark"
argc=$#
if [ $argc -eq 0 ]; then
    backend="spark"
else
    backend=$1
fi
echo "Start Orca NCF tutorial Test - $backend backend"

# download dataset from ftp
rm -f ./orca-tutorial-ncf-dataset-compressed.zip
rm -rf ml-1m
wget $FTP_URI/analytics-zoo-data/orca-tutorial-ncf-dataset-compressed.zip
unzip orca-tutorial-ncf-dataset-compressed.zip
echo "Successfully got dataset from ftp"

echo "#1 Running pytorch_train_dataloader"
#timer
start=$(date "+%s")

# ml-1m is the default dataset, thus arg --data-dir is unnecessary
python ./pytorch_train_dataloader.py --backend $backend
# pytorch dataloader does not have predict
python ./pytorch_resume_train_dataloader.py --backend $backend

now=$(date "+%s")
time1=$((now - start))

clean

echo "#2 Running pytorch_train_spark_dataframe"
#timer
start=$(date "+%s")

python ./pytorch_train_spark_dataframe.py --backend $backend
python ./pytorch_predict_spark_dataframe.py --backend $backend
python ./pytorch_resume_train_spark_dataframe.py --backend $backend

now=$(date "+%s")
time2=$((now - start))

clean

echo "#3 Running pytorch_train_xshards"
#timer
start=$(date "+%s")

python ./pytorch_train_xshards.py --backend $backend
python ./pytorch_predict_xshards.py --backend $backend
python ./pytorch_resume_train_xshards.py --backend $backend

now=$(date "+%s")
time3=$((now - start))

clean

echo "#4 Running tf_train_spark_dataframe"
#timer
start=$(date "+%s")

python ./tf_train_spark_dataframe.py --backend $backend
python ./tf_predict_spark_dataframe.py --backend $backend
python ./tf_resume_train_spark_dataframe.py --backend $backend

now=$(date "+%s")
time4=$((now - start))

clean

echo "#5 Running tf_train_xshards"
#timer
start=$(date "+%s")

python ./tf_train_xshards.py --backend $backend
python ./tf_predict_xshards.py --backend $backend
python ./tf_resume_train_xshards.py --backend $backend

now=$(date "+%s")
time5=$((now - start))

clean

echo "#1 Running pytorch_train_dataloader time used: $time1 seconds"
echo "#2 Running pytorch_train_spark_dataframe time used: $time2 seconds"
echo "#3 Running pytorch_train_xshards time used: $time3 seconds"
echo "#4 Running tf_train_spark_dataframe time used: $time4 seconds"
echo "#5 Running tf_train_xshards time used: $time5 seconds"

#clean dataset
rm -rf ml-1m
rm -f orca-tutorial-ncf-dataset-compressed.zip

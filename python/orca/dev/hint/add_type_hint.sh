#!/usr/bin/env bash

#
# Copyright 2022 The BigDL Authors.
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


# set -ex

cd "`dirname $0`"
export MT_DB_PATH="$(pwd)/orca_hint.sqlite3"
echo $MT_DB_PATH

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

ray stop -f

cd ../../
echo "Automatically Add Type Hint"

test_dir_name="data"
if [ "$1" ]; then
    test_dir_name=$1
    echo test_dir_name:${test_dir_name}
fi


if [ -f $MT_DB_PATH ];then
    rm $MT_DB_PATH
fi

for file in $(find test/bigdl/orca/${test_dir_name}  -maxdepth 1 -name test_*.py|grep -wv test_read_parquet_images.py)
do
    echo $file
    monkeytype run $file
done

cd -
add_hint_modules=$(monkeytype list-modules |grep bigdl.orca.${test_dir_name}|grep -wv test)

# for module in $add_hint_modules
# do
#     monkeytype apply $module
# done

# if [ -f $MT_DB_PATH ];then
#     rm $MT_DB_PATH
# fi
unset MT_DB_PATH

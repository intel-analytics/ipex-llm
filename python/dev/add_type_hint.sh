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


hint_module_name="friesian"
if [ "$1" ]; then
    hint_module_name=$1
    echo hint_module_name:${hint_module_name}
fi

hint_submodule_name="feature"
if [ "$2" ]; then
    hint_submodule_name=$2
    echo hint_submodule_name:${hint_submodule_name}
fi

cd "`dirname $0`"
export MT_DB_PATH="$(pwd)/${hint_module_name}_hint.sqlite3"
echo $MT_DB_PATH

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

cd ../${hint_module_name}
echo "Automatically Add Type Hint"


if [ -f $MT_DB_PATH ];then
    rm $MT_DB_PATH
fi

for file in $(find test/bigdl/${hint_module_name}/${hint_submodule_name} -name test_*.py)
do
    echo $file
    monkeytype run $file
done

cd -
unset MT_DB_PATH

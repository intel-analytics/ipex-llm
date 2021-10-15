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

RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
NANO_DIR="$(cd ${RUN_SCRIPT_DIR}/../; pwd)"
echo $NANO_DIR
cd $NANO_DIR

wheel_command="python setup.py bdist_wheel --plat-name manylinux2010_x86_64"
echo "Packing python distribution:   $wheel_command"
${wheel_command}

upload_command="twine upload dist/bigdl_nano-0.14.0.dev0-py3-none-manylinux2010_x86_64.whl"
echo "Please manually upload with this command:  $upload_command"

# $upload_command

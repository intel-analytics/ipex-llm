#!/bin/bash

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

set -x
cd /ppml
export JUPYTER_RUNTIME_DIR=/ppml/jupyter/runtime
export JUPYTER_DATA_DIR=/ppml/jupyter/data

if [ ! -d $JUPYTER_RUNTIME_DIR ]
then
  mkdir -p $JUPYTER_RUNTIME_DIR
fi

if [ ! -d $JUPYTER_DATA_DIR ]
then
  mkdir -p $JUPYTER_DATA_DIR
fi

export SECURE_PASSWORD=`openssl rsautl -inkey /ppml/password/key.txt -decrypt < /ppml/password/output.bin`
export RUNTIME_DRIVER_HOST=$( hostname -I | awk '{print $1}' )
if [ "$SGX_ENABLED" == "true" ]
then
  bash init.sh
  echo "export JUPYTER_RUNTIME_DIR=$JUPYTER_RUNTIME_DIR && \
    export JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR && \
    export SECURE_PASSWORD=$SECURE_PASSWORD && \
    export RUNTIME_DRIVER_HOST=$RUNTIME_DRIVER_HOST && \
    /usr/local/bin/jupyter-lab notebook \
    --notebook-dir=/ppml/apps \
    --ip=0.0.0.0 \
    --port=$JUPYTER_PORT \
    --no-browser \
    --NotebookApp.token=$SECURE_PASSWORD \
    --allow-root" >> temp_command_file
  export sgx_command="bash temp_command_file"
  gramine-sgx bash 2>&1 | tee /ppml/jupyter-notebook.log
else
  /usr/local/bin/jupyter-lab notebook --notebook-dir=/ppml/apps --ip=0.0.0.0 --port=$JUPYTER_PORT --no-browser --NotebookApp.token=$SECURE_PASSWORD --allow-root
fi

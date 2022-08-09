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
if [[ -z "${NOTEBOOK_PORT}" || -z "${NOTEBOOK_TOKEN}" ]]
then 
    echo "NOTEBOOK_TOKEN and NOTEBOOK_PORT cannot be empty!"
    exit 1
else
    echo $BIGDL_HOME
    jupyter-lab --notebook-dir=$BIGDL_HOME/apps --ip=0.0.0.0 --port=$NOTEBOOK_PORT --no-browser --NotebookApp.token=$NOTEBOOK_TOKEN --allow-root
fi


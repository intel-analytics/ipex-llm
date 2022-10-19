#!/usr/bin/env bash

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

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
echo "SCRIPT_DIR": $SCRIPT_DIR
export DLLIB_PYTHON_HOME="$(cd ${SCRIPT_DIR}/../../dllib/src; pwd)"
export PPML_PYTHON_HOME="$(cd ${SCRIPT_DIR}/../src; pwd)"
export BIGDL_HOME="$(cd ${SCRIPT_DIR}/../../..; pwd)"


echo "DLLIB_PYTHON_HOME": $DLLIB_PYTHON_HOME
echo "PPML_PYTHON_HOME": $PPML_PYTHON_HOME

export PYTHONPATH=$PYTHONPATH:$DLLIB_PYTHON_HOME:$PPML_PYTHON_HOME:$PPML_PYTHON_HOME/bigdl/ppml/fl/nn/generated
echo "PYTHONPATH": $PYTHONPATH

export BIGDL_CLASSPATH=$(find $BIGDL_HOME/dist/lib/ -name "bigdl-ppml*with-dependencies.jar" | head -n 1)
echo "BIGDL_CLASSPATH": $BIGDL_CLASSPATH

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

#!/bin/bash
set -e
RUN_SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

DEV_DIR="$(cd ${RUN_SCRIPT_DIR}/../dev; pwd)"
echo "DEV_DIR": $DEV_DIR
cd "`dirname $0`"

. $DEV_DIR/prepare_env.sh

red=`tput setaf 1`
green=`tput setaf 2`
cyan=`tput setaf 6`
reset=`tput sgr0`
PYTHON_EXECUTABLES=("python2" "python3")

function cleanup {
    find ${DL_PYTHON_HOME} -name "tmp_*.ipynb" -exec rm {} \;
    tput sgr0
}

function start_integration () {
    for command in $(find . -name "run-local*.sh")
    do
        printf "${green}Running : $command ${green}"
        start=`date +%s`
        $command > run-all-local.log
        if (( $? )); then
            printf " ${red}Failed. \nPlease check ${DEV_DIR}/../local_integration/run-all-local.log\n${reset}";
            exit 1;
        fi
        end=`date +%s`
        printf "  [Succeeded! Took $((end-start)) secs] \n"
    done

    find ${RUN_SCRIPT_DIR} -name "run-all-local.log" -exec rm {} \;
}

trap cleanup EXIT

for p in ${PYTHON_EXECUTABLES[@]}
do
    echo "${cyan}Using python version: $p${reset}"
    export PYTHON_EXECUTABLE=$p
    export PYSPARK_PYTHON=$p
    start_integration
done

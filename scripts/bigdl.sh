#!/bin/sh

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

#
# This script is to set right environment variables for BigDL library. User can use this script to
# invoke command(e.g. spark-submit), like this
#     bigdl.sh [options] -- command
# Note that the command should be placed behind the -- symbol. And the options of bigdl should be
# placed before the -- symbol.
#
# User can also source this script, then can run command without the script. Like this
#     source bigdl.sh [options]
#     command1
#     command2
#
# Options:
#   -c : core number to run program. Note that it will be override by Spark configuration, like
#        n of local[n] in Spark local mode, or --executor-cores in Spark cluster mode. So it only
#        takes affect when program don't run on Spark.
#

CMD=""
EXIT_CODE=0

set_mkl_blas() {
    # TODO: Generate export statements in the build from spark-bigdl.conf
    export OMP_NUM_THREADS=1
    export KMP_BLOCKTIME=0
    export OMP_WAIT_POLICY=passive
    export DL_ENGINE_TYPE=mklblas
    export MKL_DISABLE_FAST_MM=1
}

parse() {
    while [ $# -gt 0 ]; do
        key="$1"
        case $key in
            -c|--core)
            export DL_CORE_NUMBER="$2"
            shift 2
            ;;
            -l|--local)
            export BIGDL_LOCAL_MODE="1"
            shift 1
            ;;
            --)
            shift
            for token in "$@"; do
                case "$token" in
                    *\ * )
                    # add quto when there's space in the token
                    CMD="$CMD '$token'"
                    ;;
                    *)
                    CMD="$CMD $token"
                    ;;
                esac
            done
            shift $#    # escape from the while loop
            ;;
            *)
            echo "Invalid option $key"
            shift $#    # escape from the while loop
            EXIT_CODE=1
            ;;
        esac
    done
}

parse "$@"

if [ $EXIT_CODE -ne 0 ]; then
    return $EXIT_CODE
fi

# set the envs based on the mode
set_mkl_blas

if [ "$CMD" != "" ]; then
    eval $CMD
fi

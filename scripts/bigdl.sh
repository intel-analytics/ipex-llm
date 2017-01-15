#!/bin/sh

#
# Licensed to Intel Corporation under one or more                                                                   
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set_mkl_blas() {
    echo "Set engine type to mkl_blas"
    export OMP_NUM_THREADS=1
    export KMP_BLOCKTIME=0
    export OMP_WAIT_POLICY=passive
    export DL_ENGINE_TYPE=mklblas
}

if [ -z "$1" ]
then
    set_mkl_blas
else
    if [ "$1" = "mkl_blas" ]; then
        set_mkl_blas
        if [ "$2" = "--" ]; then
            shift 2
            eval $*
        fi
    elif [ "$1" = "--" ]; then
        set_mkl_blas
        shift
        eval $*
    else
        echo "Invalid engine type '$1'. Only support mkl_blas now"
    fi
fi


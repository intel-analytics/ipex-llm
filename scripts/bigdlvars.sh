#!/bin/sh

if [ -z "$1" ]
then
    echo "No engine type"
else
    if [ "$1" = "mkl_blas" ]; then
        echo "Set engine type to mkl_blas"
        export OMP_NUM_THREADS=1
        export KMP_BLOCKTIME=0
        export OMP_WAIT_POLICY=passive
        export DL_ENGINE_TYPE=mklblas
    elif [ "$1" = "mkl_dnn" ]; then
        echo "Set engine type to mkl_dnn"
        unset OMP_NUM_THREADS
        unset KMP_BLOCKTIME
        unset OMP_WAIT_POLICY
        export DL_ENGINE_TYPE=mkldnn
    else
        echo "Invalid engine type"
    fi
fi

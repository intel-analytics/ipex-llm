#!/bin/bash

set -e

echo "Start ray horovod pytorch example tests"
#start execute
echo "#1 pytorch estimator example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/learn/horovod/pytorch_estimator.py
now=$(date "+%s")
time1=$((now-start))
echo "horovod pytorch example tests finished"

echo "#1 pytorch estimator example time used:$time1 seconds"

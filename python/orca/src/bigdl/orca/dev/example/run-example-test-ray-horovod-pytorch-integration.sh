#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "#1 Start ray horovod pytorch example tests"
start=$(date "+%s")
# run example
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/horovod/pytorch_estimator.py --cluster_mode yarn --num_nodes 2 
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "ray horovod pytorch failed"
    exit $exit_status
fi

now=$(date "+%s")
time1=$((now-start))

# This should be done at the very end after all tests finish. Or put the uninstalling at jenkins configure.
# clear_up

echo "#1 pytorch estimator example time used:$time1 seconds"


#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}

echo "#11 start test for orca transfer learning"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca transfer learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#11 Total time cost ${time} seconds"

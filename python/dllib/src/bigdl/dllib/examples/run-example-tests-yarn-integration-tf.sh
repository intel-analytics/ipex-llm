#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}


echo "#15 start test for orca tf transfer_learning "
#timer   
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py --batch_size 32 --cluster_mode yarn-client --data_dir /data/diankun
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca torchmodel mnist failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#15 Total time cost ${time} seconds"

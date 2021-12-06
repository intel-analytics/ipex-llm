#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}


echo "#11 start test for orca transfer learning"
#timer
start=$(date "+%s")
#run the example
cat /etc/hostname

python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py \
  --batch_size 4 \
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

: '

echo "#11 start test for orca transfer learning"
#timer
start=$(date "+%s")
#run the example
cat /etc/hostname

python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py \
  --batch_size 4 \
  --data_dir /opt/work/manfei/BigDL/python/orca/example/learn/tf/transfer_learning/cats_and_dogs_filtered \
  --cluster_mode local
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca transfer learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#11 Total time cost ${time} seconds"

'

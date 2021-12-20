#!/bin/bash


echo "#3 start test for orca inception inception"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/inception/inception.py  \
  --imagenet /data/imagenettfrecord \
  --py_files ${BIGDL_ROOT}/python/orca/example/learn/tf/inception/inception_preprocessing.py \
  -b 128 --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca inception failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#3 Total time cost ${time} seconds"

echo "#7 start test for orca inception inception"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/inception/inception.py  \
  --imagenet /data/imagenettfrecord \
  --py_files ${BIGDL_ROOT}/python/orca/example/learn/tf/inception/inception_preprocessing.py \
  -b 128 --cluster_mode yarn-cluster
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca inception failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#3 Total time cost ${time} seconds"
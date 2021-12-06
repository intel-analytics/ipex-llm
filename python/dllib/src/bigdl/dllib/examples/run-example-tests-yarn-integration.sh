#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}

echo "#12 start test for orca tf2 resnet-50-imagenet"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf2/resnet/resnet-50-imagenet.py  \
  --worker_num 8 --cores 17 \
  --data_dir /data/imagenettfrecord --use_bf16 \
  --enable_numa_binding \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tf2 resnet-50-imagenet failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#12 Total time cost ${time} seconds"

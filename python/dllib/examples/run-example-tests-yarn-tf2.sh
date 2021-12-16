#!/bin/bash

# echo "#9 start test for orca learn tf2 resnet"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf2/resnet/resnet-50-imagenet.py  \
#   --worker_num 2 \
#   --cores 32 \
#   --memory "20g" \
#   --data_dir /data/imagenettfrecord/train \
#   --num_images_train 1281 \
#   --num_images_validation 50 \
#   --epochs 2 \
#   --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca learning learn tf2 resnet failed"
#   #exit $exit_status
# fi
# now==$(date "+%s")
# time==$((now - start))
# echo "#9 Total time cost ${time} seconds"

# echo "#9 start test for orca learn tf2 resnet"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf2/resnet/resnet-50-imagenet.py  \
#   --worker_num 2 \
#   --cores 32 \
#   --memory "20g" \
#   --data_dir /data/imagenettfrecord/train \
#   --num_images_train 1281 \
#   --num_images_validation 50 \
#   --epochs 2 \
#   --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca learning learn tf2 resnet failed"
#   #exit $exit_status
# fi
# now==$(date "+%s")
# time==$((now - start))
# echo "#9 Total time cost ${time} seconds"

echo "#23 start test for orca yolov3 yoloV3"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf2/yolov3/yoloV3.py  \
  --data_dir /data/yolov3 \
  --output_data /data/yolov3/parquet \
  --weights /data/yolov3/yolov3.weights \
  --names /data/yolov3/voc2012.names \
  --epochs 1 --cluster_mode yarn-client \
  --memory 20g \
  --object_store_memory 10g \
  --batch_size 64 \
  --cores 8 \
  --checkpoint /data/checkpoints/yolov3.tf \
  --checkpoint_folder /data/checkpoints
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "orca yolov3 failed"
  #exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#23 Total time cost ${time} seconds"

echo "#23 start test for orca yolov3 yoloV3"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf2/yolov3/yoloV3.py  \
  --data_dir /data/yolov3 \
  --output_data /data/yolov3/parquet \
  --weights /data/yolov3/yolov3.weights \
  --names /data/yolov3/voc2012.names \
  --epochs 1 --cluster_mode yarn-cluster \
  --memory 20g \
  --object_store_memory 10g \
  --batch_size 64 \
  --cores 8 \
  --checkpoint /data/checkpoints/yolov3.tf \
  --checkpoint_folder /data/checkpoints
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "orca yolov3 failed"
  #exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#23 Total time cost ${time} seconds"

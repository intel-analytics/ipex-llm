#!/bin/bash

echo "#1-1 start test for orca tf basic_text_classification"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/basic_text_classification.py --cluster_mode yarn-client --data_dir /data/imdb
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca basic_text_classification failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#1-1 Total time cost ${time} seconds"

echo "#1-2 start test for orca tf basic_text_classification basic_text_classification"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/basic_text_classification.py --cluster_mode yarn-cluster --data_dir /data/imdb
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca basic_text_classification failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#1-2 Total time cost ${time} seconds"

echo "#2-1 start test for orca tf image_segmentation.py"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/image_segmentation/image_segmentation.py \
  --batch_size 64 \
  --file_path /data/carvana \
  --non_interactive --epochs 1 --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca image_segmentation failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#2-1 Total time cost ${time} seconds"

echo "#2-2 start test for orca tf image_segmentation image_segmentation.py"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/image_segmentation/image_segmentation.py \
  --batch_size 64 \
  --file_path /data/carvana \
  --non_interactive --epochs 1 --cluster_mode yarn-cluster
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca image_segmentation failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#2-2 Total time cost ${time} seconds"

echo "#3-1 start test for orca inception inception"
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
echo "#3-1 Total time cost ${time} seconds"

echo "#3-2 start test for orca inception inception"
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
echo "#3-2 Total time cost ${time} seconds"

echo "#4-1 start test for orca learn transfer_learning"
#timer 
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py --data_dir /data --cluster_mode yarn-client --download False
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca learning transfer_learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#4-1 Total time cost ${time} seconds"

# echo "#4-2 start test for orca learn transfer_learning"
# #timer 
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py --data_dir /data --cluster_mode yarn-cluster --download False
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   echo "orca learning transfer_learning failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#4-2 Total time cost ${time} seconds"
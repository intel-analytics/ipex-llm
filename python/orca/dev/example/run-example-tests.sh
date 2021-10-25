#!/bin/bash

set -e

# echo "#1 start example test for openvino"
# #timer
# start=$(date "+%s")
# if [ -f models/faster_rcnn_resnet101_coco.xml ]; then
#   echo "models/faster_rcnn_resnet101_coco already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.xml \
#     -P models
#   wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.bin \
#     -P models
# fi
# if [ -d tmp/data/object-detection-coco ]; then
#   echo "tmp/data/object-detection-coco already exists"
# else
#   wget -nv $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P tmp/data
#   unzip -q tmp/data/object-detection-coco.zip -d tmp/data
# fi
# python ${BIGDL_ROOT}/python/orca/example/openvino/predict.py \
#   --image tmp/data/object-detection-coco \
#   --model models/faster_rcnn_resnet101_coco.xml
# now=$(date "+%s")
# time1=$((now - start))

# echo "#2 start example for vnni/openvino"
# #timer
# start=$(date "+%s")
# if [ -d models/vnni ]; then
#   echo "models/resnet_v1_50.xml already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-models/openvino/vnni/resnet_v1_50.zip \
#     -P models
#   unzip -q models/resnet_v1_50.zip -d models/vnni
# fi
# if [ -d tmp/data/object-detection-coco ]; then
#   echo "tmp/data/object-detection-coco already exists"
# else
#   wget -nv $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P tmp/data
#   unzip -q tmp/data/object-detection-coco.zip -d tmp/data
# fi
# python ${BIGDL_ROOT}/python/orca/example/vnni/openvino/predict.py \
#   --model models/vnni/resnet_v1_50.xml \
#   --image tmp/data/object-detection-coco
# now=$(date "+%s")
# time2=$((now - start))

# echo "#3 start example test for tensorflow"
# #timer
# start=$(date "+%s")
# if [ -f models/resnet_50_saved_model.zip ]; then
#   echo "-models/resnet_50_saved_model.zip already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-models/tensorflow/resnet_50_saved_model.zip \
#     -P models
#   unzip models/resnet_50_saved_model.zip -d models/resnet_50_saved_model
# fi

# echo "start example test for TFPark freeze saved model 9"
# python ${BIGDL_ROOT}/python/orca/example/freeze_saved_model/freeze.py \
#   --saved_model_path models/resnet_50_saved_model \
#   --output_path models/resnet_50_tfnet

# now=$(date "+%s")
# time3=$((now - start))

# echo "#4 start example test for orca data"
# if [ -f tmp/data/NAB/nyc_taxi/nyc_taxi.csv ]; then
#   echo "tmp/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
# else
#   wget -nv $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
#     -P tmp/data/NAB/nyc_taxi/
# fi
# #timer
# start=$(date "+%s")
# python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py \
#   -f tmp/data/NAB/nyc_taxi/nyc_taxi.csv

# now=$(date "+%s")
# time4=$((now - start))

# echo "#5 start test for orca tf imagesegmentation"
# #timer
# start=$(date "+%s")
# # prepare data
# if [ -f tmp/data/carvana ]; then
#   echo "tmp/data/carvana already exists"
# else
#   wget $FTP_URI/analytics-zoo-data/data/carvana/train.zip \
#     -P tmp/data/carvana/
#   wget $FTP_URI/analytics-zoo-data/data/carvana/train_masks.zip \
#     -P tmp/data/carvana/
#   wget $FTP_URI/analytics-zoo-data/data/carvana/train_masks.csv.zip \
#     -P tmp/data/carvana/
# fi

# # Run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/image_segmentation/image_segmentation.py \
#   --file_path tmp/data/carvana --epochs 1 --non_interactive
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   echo "orca tf imagesegmentation failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time5=$((now - start))

echo "#6 start test for orca tf transfer_learning"
#timer
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca tf transfer_learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time6=$((now - start))

echo "#7 start test for orca tf basic_text_classification"
#timer
start=$(date "+%s")
sed "s/epochs=100/epochs=10/g" \
  ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/basic_text_classification.py \
  >${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/tmp.py
python ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/tmp.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca tf basic_text_classification failed"
  exit $exit_status
fi
now=$(date "+%s")
time7=$((now - start))

echo "#8 start test for orca bigdl attention"
#timer
start=$(date "+%s")
sed "s/max_features = 20000/max_features = 200/g;s/max_len = 200/max_len = 20/g;s/hidden_size=128/hidden_size=8/g;s/memory=\"100g\"/memory=\"20g\"/g;s/driver_memory=\"20g\"/driver_memory=\"3g\"/g" \
  ${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/transformer.py \
  >${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/tmp.py
python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/tmp.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca tf bigdl attention failed"
  exit $exit_status
fi
now=$(date "+%s")

time8=$((now - start))

echo "#9 start test for orca bigdl imageInference"
#timer
start=$(date "+%s")
if [ -f models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
  echo "models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
  wget -nv $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P models
fi

python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/imageInference/imageInference.py \
  -m models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/kaggle/train_100
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca bigdl imageInference failed"
  exit $exit_status
fi
now=$(date "+%s")
time9=$((now - start))

echo "#1 openvino time used: $time2 seconds"
echo "#2 vnni/openvino time used: $time3 seconds"
echo "#3 tensorflow time used: $time4 seconds"
echo "#4 orca data time used:$time5 seconds"
#echo "#5 orca tf imagesegmentation time used:$time6 seconds"
echo "#6 orca tf transfer_learning time used:$time7 seconds"
echo "#7 orca tf basic_text_classification time used:$time8 seconds"
echo "#8 orca bigdl attention time used:$time9 seconds"
echo "#9 orca bigdl imageInference time used:$time10 seconds"



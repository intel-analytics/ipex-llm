#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH
export BIGDL_JARS=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

set -e
echo "Start ray horovod tf example tests"

echo "#1 tf2 estimator resnet 50 example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf2/resnet/resnet-50-imagenet.py --use_dummy_data --benchmark --batch_size_per_worker 4
now=$(date "+%s")
time1=$((now-start))

echo "Ray example tests finished"

echo "Start yoloV3 tf2 example tests"

echo "#2 tf2 estimator yoloV3 example"
start=$(date "+%s")
#if [ -f analytics-zoo-models/yolov3.weights ]; then
#  echo "analytics-zoo-models/yolov3.weights already exists."
#else
#  wget -nv $FTP_URI/analytics-zoo-models/yolov3/yolov3.weights \
#    -P analytics-zoo-models
#fi

#if [ -f analytics-zoo-data/voc2012.names ]; then
#  echo "analytics-zoo-data/voc2012.names already exists."
#else
#  wget -nv $FTP_URI/analytics-zoo-data/yolov3/voc2012.names -P analytics-zoo-data
#fi

if [ -f analytics-zoo-data/coco.names ]; then
  echo "analytics-zoo-data/coco.names already exists."
else
  wget -nv $FTP_URI/analytics-zoo-data/yolov3/coco.names -P analytics-zoo-data
fi

if [ -f analytics-zoo-data/VOCdevkit.zip ]; then
  echo "analytics-zoo-data/VOCdevkit.zip already exists."
else
  wget -nv $FTP_URI/analytics-zoo-data/yolov3/VOCdevkit.zip -P analytics-zoo-data
  unzip -q analytics-zoo-data/VOCdevkit.zip -d analytics-zoo-data/VOCdevkit
fi

if [ -f analytics-zoo-models/checkpoints.zip ]; then
  echo "analytics-zoo-models/checkpoints already exists."
else
  wget -nv $FTP_URI/analytics-zoo-models/yolov3/checkpoints.zip \
    -P analytics-zoo-models
  unzip -q analytics-zoo-models/checkpoints.zip -d analytics-zoo-models
fi

#echo "yolov3 train"
#disable test due to small /dev/shm shared memory on jenkins
#python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf2/yolov3/yoloV3.py --data_dir analytics-zoo-data/VOCdevkit --weights analytics-zoo-models/yolov3.weights --class_num 20 --names analytics-zoo-data/voc2012.names --data_year 2007 --split_name_train trainval --split_name_test trainval --object_store_memory 1g

echo "yolov3 predict"
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf2/yolov3/predict.py --checkpoint analytics-zoo-models/checkpoints/yolov3.tf --names analytics-zoo-data/coco.names --class_num 80 --image analytics-zoo-data/VOCdevkit/VOCdevkit/VOC2007/JPEGImages/000005.jpg

now=$(date "+%s")
time2=$((now-start))

echo "#1 tf2 estimator resnet 50 time used:$time1 seconds"
echo "#2 tf2 estimator yolov3 time used:$time2 seconds"


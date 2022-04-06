#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling BigDL"
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

set -e

ray stop -f
ray start --head

echo "#start orca tf2 example tests"
echo "#1 tf2 estimator resnet 50 example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/learn/tf2/resnet/resnet-50-imagenet.py \
  --runtime ray --use_dummy_data --benchmark --batch_size_per_worker 4
now=$(date "+%s")
time1=$((now-start))

ray stop -f
#ray start --head

# echo "#2 tf2 estimator resnet 50 example"
# if [ -f tmp/coco.names ]; then
#   echo "tmp/coco.names already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-data/yolov3/coco.names -P tmp
# fi

# if [ -f tmp/VOCdevkit.zip ]; then
#   echo "tmp/VOCdevkit.zip already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-data/yolov3/VOCdevkit.zip -P tmp
#   unzip -q tmp/VOCdevkit.zip -d tmp/VOCdevkit
# fi

# if [ -f models/checkpoints.zip ]; then
#   echo "models/checkpoints already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-models/yolov3/checkpoints.zip \
#     -P models
#   unzip -q models/checkpoints.zip -d models
# fi

# echo "yolov3 predict"
# start=$(date "+%s")
# python ${BIGDL_ROOT}/python/orca/example/learn/tf2/yolov3/predict.py \
#   --checkpoint models/checkpoints/yolov3.tf \
#   --names tmp/coco.names --class_num 80 \
#   --image tmp/VOCdevkit/VOCdevkit/VOC2007/JPEGImages/000005.jpg

# now=$(date "+%s")
# time2=$((now-start))

# ray stop -f

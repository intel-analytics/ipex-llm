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

# echo "#2 tf2 estimator yolov3 example"
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
ray start --head

echo "#start orca pytorch example tests"
echo "#3 start example for orca fashion-mnist"
start=$(date "+%s")
if [ -d ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/data ]
then
    echo "fashion-mnist already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/fashion-mnist.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/
    unzip ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion-mnist.zip
fi
sed "s/epochs=5/epochs=1/g;s/batch_size=4/batch_size=256/g" \
    ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py \
    > ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py \
    --runtime ray --address localhost:6379 --backend ray --batch_size=256
now=$(date "+%s")
time3=$((now-start))

echo "#4 start example for orca cifar10"
start=$(date "+%s")
if [ -d ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/data ]; then
  echo "Cifar10 already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10
  unzip ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.zip
fi

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py \
    --runtime ray --address localhost:6379 --backend ray --batch_size 256 --epochs 1
now=$(date "+%s")
time4=$((now-start))

echo "#5 start example for orca super-resolution"
start=$(date "+%s")
if [ ! -f BSDS300-images.tgz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/BSDS300-images.tgz
fi
if [ ! -d dataset/BSDS300/images ]; then
  mkdir dataset
  tar -xzf BSDS300-images.tgz -C dataset
fi
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py \
  --runtime ray --address localhost:6379 --backend ray
now=$(date "+%s")
time5=$((now-start))

ray stop -f

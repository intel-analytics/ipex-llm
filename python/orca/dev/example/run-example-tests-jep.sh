#!/bin/bash
# TODO: remove jep
set -e

echo "#1 start example for MNIST"
#timer
start=$(date "+%s")
if [ -f tmp/data/MNIST ]; then
  echo "MNIST already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P tmp/data/MNIST/raw
  wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P tmp/data/MNIST/raw
  wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P tmp/data/MNIST/raw
  wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P tmp/data/MNIST/raw
fi

python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py --dir tmp/data --epochs 1

now=$(date "+%s")
time1=$((now - start))

echo "#5 start test for orca bigdl resnet-finetune"
#timer
start=$(date "+%s")
#prepare dataset
wget $FTP_URI/analytics-zoo-data/data/cats_and_dogs_filtered.zip -P tmp/data
unzip -q tmp/data/cats_and_dogs_filtered.zip -d tmp/data
mkdir tmp/data/cats_and_dogs_filtered/samples
cp tmp/data/cats_and_dogs_filtered/train/cats/cat.7* tmp/data/cats_and_dogs_filtered/samples
cp tmp/data/cats_and_dogs_filtered/train/dogs/dog.7* tmp/data/cats_and_dogs_filtered/samples
#prepare model
if [ -d ${HOME}/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth ]; then
  echo "resnet model found."
else
  if [ ! -d ${HOME}/.cache/torch/hub/checkpoints ]; then
    mkdir ${HOME}/.cache/torch/hub/checkpoints
  fi
  wget $FTP_URI/analytics-zoo-models/pytorch/resnet18-5c106cde.pth -P ${HOME}/.cache/torch/hub/checkpoints
fi
#run the example
python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/resnet_finetune/resnet_finetune.py tmp/data/cats_and_dogs_filtered/samples
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca bigdl resnet-finetune"
  exit $exit_status
fi
now=$(date "+%s")
time5=$((now - start))

echo "#1 MNIST example time used:$time1 seconds"
echo "#2 orca Cifar10 example time used:$time2 seconds"
echo "#3 orca Fashion-MNIST example time used:$time3 seconds"
echo "#4 orca Super Resolution example time used:$time4 seconds"
echo "#5 torchmodel resnet-finetune time used:$time5 seconds"
echo "#6 orca brainMRI example time used:$time6 seconds"

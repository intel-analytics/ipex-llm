#!/bin/bash

set -e

echo "#1 start example for MNIST"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/MNIST ]; then
  echo "MNIST already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
  wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
  wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
  wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
fi

python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py --dir analytics-zoo-data/data

now=$(date "+%s")
time1=$((now - start))

echo "#2 start example for orca Cifar10"
#timer
start=$(date "+%s")
if [ -d ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/data ]; then
  echo "Cifar10 already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10
  unzip ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.zip
fi

sed "s/epochs=2/epochs=1/g;s/batch_size=4/batch_size=256/g" \
  ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py \
  >${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10_tmp.py

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10_tmp.py

now=$(date "+%s")
time2=$((now - start))

echo "#3 start example for orca Fashion-MNIST"
#timer
start=$(date "+%s")
if [ -d ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/data ]
then
    echo "fashion-mnist dataset already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/fashion-mnist.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/
    unzip ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion-mnist.zip
fi

sed "s/epochs=5/epochs=1/g;s/batch_size=4/batch_size=256/g" \
    ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py \
    > ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py

now=$(date "+%s")
time3=$((now - start))

echo "#4 start example for orca Super Resolution"
#timer
start=$(date "+%s")
if [ ! -f BSDS300-images.tgz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/BSDS300-images.tgz
fi
if [ ! -d dataset/BSDS300/images ]; then
  mkdir dataset
  tar -xzf BSDS300-images.tgz -C dataset
fi

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py

now=$(date "+%s")
time4=$((now - start))

# echo "#3 start test for orca bigdl resnet-finetune"
# #timer
# start=$(date "+%s")
# #prepare dataset
# wget $FTP_URI/analytics-zoo-data/data/cats_and_dogs_filtered.zip -P analytics-zoo-data/data
# unzip -q analytics-zoo-data/data/cats_and_dogs_filtered.zip -d analytics-zoo-data/data
# mkdir analytics-zoo-data/data/cats_and_dogs_filtered/samples
# cp analytics-zoo-data/data/cats_and_dogs_filtered/train/cats/cat.7* analytics-zoo-data/data/cats_and_dogs_filtered/samples
# cp analytics-zoo-data/data/cats_and_dogs_filtered/train/dogs/dog.7* analytics-zoo-data/data/cats_and_dogs_filtered/samples
# #prepare model
# if [ -d ${HOME}/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth ]; then
#   echo "resnet model found."
# else
#   if [ ! -d ${HOME}/.cache/torch/hub/checkpoints ]; then
#     mkdir ${HOME}/.cache/torch/hub/checkpoints
#   fi
#   wget $FTP_URI/analytics-zoo-models/pytorch/resnet18-5c106cde.pth -P ${HOME}/.cache/torch/hub/checkpoints
# fi
# #run the example
# python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/resnet_finetune/resnet_finetune.py --imagePath analytics-zoo-data/data/cats_and_dogs_filtered/samples
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca bigdl resnet-finetune"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time3=$((now - start))

echo "#1 MNIST example time used:$time1 seconds"
echo "#2 orca Cifar10 example time used:$time2 seconds"
echo "#3 orca Fashion-MNIST example time used:$time3 seconds"
echo "#4 orca Super Resolution example time used:$time4 seconds"
#echo "#3 orca bigdl resnet-finetune time used:$time3 seconds"

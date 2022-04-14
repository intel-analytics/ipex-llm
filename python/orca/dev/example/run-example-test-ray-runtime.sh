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

echo "#start orca pytorch example tests"
echo "#1 start example for orca fashion-mnist"
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
time1=$((now-start))

echo "#2 start example for orca cifar10"
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
time2=$((now-start))

ray stop -f
ray start --head

echo "#3 start example for orca super-resolution"
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
time3=$((now-start))

ray stop -f

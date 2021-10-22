#!/usr/bin/env bash
clear_up() {
  echo "Clearing up environment. Uninstalling bigdl-dllib"
  pip uninstall -y bigdl-dllib
  pip uninstall -y bigdl
  pip uninstall -y pyspark
}

echo "#1 start example test for keras mnist_cnn"
start=$(date "+%s")

rm -f /tmp/mnist/*
wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P /tmp/mnist
wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P /tmp/mnist
wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P /tmp/mnist
wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P /tmp/mnist

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${BIGDL_ROOT}/python/dllib/examples/keras/mnist_cnn.py --max_epoch 2
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "mnist_cnn failed"
  exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now - start))

echo "#2 start example test for keras imdb_cnn"
#timer
start=$(date "+%s")
export SPARK_DRIVER_MEMORY=10g
python ${BIGDL_ROOT}/python/dllib/examples/keras/imdb_cnn_lstm.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "imdb cnn lstm failed"
  exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now - start))

echo "#3 start example test for lenet"
#timer
start=$(date "+%s")

export SPARK_DRIVER_MEMORY=2g
python ${BIGDL_ROOT}/python/dllib/examples/lenet/lenet.py --maxEpoch 2
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "lenet failed"
  exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time3=$((now - start))

echo "#4 start example test for nnframes"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
  echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
  wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi

if [ -f analytics-zoo-data/data/dogs-vs-cats/train.zip ]; then
  echo "analytics-zoo-data/data/dogs-vs-cats/train.zip already exists."
else
  # echo "Downloading dogs and cats images"
  wget $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip \
    -P analytics-zoo-data/data/dogs-vs-cats
  unzip analytics-zoo-data/data/dogs-vs-cats/train.zip -d analytics-zoo-data/data/dogs-vs-cats
  mkdir -p analytics-zoo-data/data/dogs-vs-cats/samples
  cp analytics-zoo-data/data/dogs-vs-cats/train/cat.7* analytics-zoo-data/data/dogs-vs-cats/samples
  cp analytics-zoo-data/data/dogs-vs-cats/train/dog.7* analytics-zoo-data/data/dogs-vs-cats/samples

  mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/cats
  mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/dogs
  cp analytics-zoo-data/data/dogs-vs-cats/train/cat.71* analytics-zoo-data/data/dogs-vs-cats/demo/cats
  cp analytics-zoo-data/data/dogs-vs-cats/train/dog.71* analytics-zoo-data/data/dogs-vs-cats/demo/dogs
  # echo "Finished downloading images"
fi

export SPARK_DRIVER_MEMORY=20g

echo "start example test for nnframes imageInference"
python ${BIGDL_ROOT}/dllib/examples/nnframes/imageInference/ImageInferenceExample.py \
  -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/kaggle/train_100

exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "nnframes_imageInference failed"
  exit $exit_status
fi

echo "start example test for nnframes transfer learning"
python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
  -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f analytics-zoo-data/data/dogs-vs-cats/samples --nb_epoch 2

exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "nnframes_imageTransferLearning failed"
  exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time4=$((now - start))

echo "start example test for autograd"
#timer
start=$(date "+%s")

export SPARK_DRIVER_MEMORY=2g
python ${BIGDL_ROOT}/python/dllib/examples/autograd/custom.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "autograd-custom failed"
  exit $exit_status
fi

python ${BIGDL_ROOT}/python/dllib/examples/autograd/customloss.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "autograd_customloss failed"
  exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time5=$((now - start))

clear_up

echo "#1 mnist cnn time used: $time1 seconds"
echo "#2 imdb cnn lstm time used: $time2 seconds"
echo "#3 lenet time used: $time3 seconds"
echo "#4 nnframes time used: $time4 seconds"
echo "#5 autograd time used: $time5 seconds"

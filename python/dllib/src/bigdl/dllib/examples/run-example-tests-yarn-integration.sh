#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}


echo "start test for dllib rnn"
echo "start test for dllib custom"
echo "start test for dllib custom loss"
echo "start test for dllib imageframe inception validation"
echo "start test for dllib keras imdb bigdl backend"
echo "start test for dllib keras imdb cnn lstm"
echo "start test for dllib keras mnist cnn"
echo "start test for dllib nnframes image transfer learning"
echo "start test for dllib nnframes image inference"

echo "#1 start test for data spark_pandas"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py \
  --deploy-mode 'yarn-client'
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "data spark_pandas failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#1 Total time cost ${time} seconds"


echo "#2 start test for pytorch cifar10"
#timer
start=$(date "+%s")
#run the example
rm -rf ./data
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py \
  --cluster_mode 'yarn-client' \
  --epochs 1 \
  --batch_size 256
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "pytorch cifar10 failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#2 Total time cost ${time} seconds"


echo "#3 start test for pytorch fashion_mnist"
#timer
start=$(date "+%s")
#run the example
rm -rf ./data
rm -rf ./runs
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py \
  --cluster_mode 'yarn-client' \
  --epochs 1 \
  --batch_size 256
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "pytorch fashion_mnist failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#3 Total time cost ${time} seconds"


echo "#4 start test for pytorch super_resolution"
#timer
start=$(date "+%s")
#run the example
rm -rf /tmp/super_resolution_data
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py \
  --cluster_mode 'yarn-client'\
  --data_dir '/tmp/super_resolution_data'
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "pytorch super_resolution failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#4 Total time cost ${time} seconds"


echo "#5 start test for torchmodel imagenet"
#timer
start=$(date "+%s")
#run the example
rm -rf /tmp/imagenet
${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/imagenet /tmp/imagenet
ls /tmp/imagenet
python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/imagenet/main.py \
  /tmp/imagenet \
  --max_epochs 1 \
  --batch-size 256 \
  --deploy_mode 'yarn-client'
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "torchmodel imagenet failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#5 Total time cost ${time} seconds"


echo "#6 start test for torchmodel mnist"
#timer
start=$(date "+%s")
#run the example
rm -rf /tmp/torchmodel_mnist
python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py \
  --dir /tmp/torchmodel_mnist \
  --epochs 1 \
  --batch-size 256 \
  --deploy-mode 'yarn-client'
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "torchmodel mnist failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#6 Total time cost ${time} seconds"


echo "#7 start test for torchmodel resnet_finetune"
#timer
start=$(date "+%s")
#run the example
rm -rf /tmp/dogscats
${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/dogscats /tmp/dogscats
ls /tmp/dogscats
python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/resnet_finetune/resnet_finetune.py \
  /tmp/dogscats \
  --deploy-mode 'yarn-client'
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "torchmodel resnet_finetune failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#7 Total time cost ${time} seconds"


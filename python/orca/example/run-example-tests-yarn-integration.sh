##!/bin/bash
#echo "#11 start test for orca data spark_pandas"
##timer 
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py --deploy-mode 'yarn-client'  -f /data/nyc_taxi.csv
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca data spark_pandas failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#11 Total time cost ${time} seconds"

#echo "#12 start test for pytorch cifar10"
##timer  
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py --cluster_mode 'yarn-cluster' --epochs 1  --batch_size 256 --data_dir /data/cifar10_data --download False
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca pytorch cifar10 failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#12 Total time cost ${time} seconds"

#echo "#13 start test for pytorch fashion_mnist"
##timer 
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py --cluster_mode 'yarn-client'   --epochs 1  --batch_size 256 --download False --data_dir /data/fashion_mnist
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca pytorch fashion_mnist failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#13 Total time cost ${time} seconds"
#
#echo "#14 start test for pytorch super_resolution"
##timer 
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py  --cluster_mode 'yarn-client' --data_dir /data/dataset 
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca pytorch super_resolution failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#14 Total time cost ${time} seconds"
#
#echo "#15 start test for orca torchmodel mnist"
##timer   
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py  --deploy-mode 'yarn-client' --dir /data/test_mnist --download False
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca torchmodel mnist failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#15 Total time cost ${time} seconds"
#
#echo "#16 start test for orca torchmodel imagenet"
##timer  
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/imagenet/main.py  /data/imagenet2012 --batch-size 8 --max_epochs 1 --deploy_mode yarn-client
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca torchmodel imagenet failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#16 Total time cost ${time} seconds"
#
#echo "#17 start test for orca torchmodel resnet_finetune"
##timer
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/resnet_finetune/resnet_finetune.py /data/dogs_cats/samples --deploy-mode yarn-client
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca torchmodel resnet_finetune failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#17 Total time cost ${time} seconds"
#
#########################  cluster
#echo "##18 start test for orca data spark_pandas"
##timer 
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py --deploy-mode 'yarn-cluster'  -f /data/nyc_taxi.csv
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca data spark_pandas failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#18 Total time cost ${time} seconds"
#
#echo "#19 start test for pytorch cifar10"
##timer  
## failed jep error
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py --cluster_mode yarn-cluster --epochs 1  --batch_size 4 --data_dir /data/cifar10_data --download False --executor_memory '4g' --driver_memory '4g'
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca pytorch cifar10 failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#19 Total time cost ${time} seconds"
#
#echo "#20 start test for pytorch fashion_mnist"
##timer 
##failed jep error
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py --cluster_mode yarn-cluster   --epochs 1  --batch_size 8 --download False --data_dir /data/fashion_mnist
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca pytorch fashion_mnist failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#20 Total time cost ${time} seconds"
#
#echo "#21 start test for orca torchmodel mnist"
##timer  
##failed jep error
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py  --deploy-mode yarn-cluster --dir /data/test_mnist --download False
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca torchmodel mnist failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#21 Total time cost ${time} seconds"
#
#echo "#22 start test for pytorch super_resolution"
##timer 
##failed jep error
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py  --cluster_mode yarn-cluster --data_dir /data/dataset 
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca pytorch super_resolution failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#22 Total time cost ${time} seconds"
#
#echo "#23 start test for orca torchmodel imagenet"
##timer   
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/imagenet/main.py  /data/imagenet2012 --batch-size 8 --max_epochs 1 --deploy_mode yarn-cluster
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca torchmodel imagenet failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#23 Total time cost ${time} seconds"
#
#echo "#24 start test for orca torchmodel resnet_finetune"
##timer 
##failed jep error
#start=$(date "+%s")
##run the example
#python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/resnet_finetune/resnet_finetune.py /data/dogs_cats/samples --deploy-mode yarn-cluster
#exit_status=$?
#if [ $exit_status -ne 0 ]; then
#  clear_up
#  echo "orca torchmodel resnet_finetune failed"
#  exit $exit_status
#fi
#now=$(date "+%s")
#time=$((now - start))
#echo "#24 Total time cost ${time} seconds"
#

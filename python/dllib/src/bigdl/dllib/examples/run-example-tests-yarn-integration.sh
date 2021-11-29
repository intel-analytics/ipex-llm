#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}

# echo "#1 start test for dllib lenet5"

# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/mnist
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/mnist /tmp/mnist
# ls /tmp/mnist
# python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/lenet/lenet5.py --on-yarn -n 1
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib lenet5 failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#1 Total time cost ${time} seconds"

# echo "#2 start test for dllib inception"

# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/inception/inception.py -f ${HDFS_URI}/imagenet-mini \
# 	--batchSize 128 \
# 	--learningRate 0.065 \
# 	--weightDecay 0.0002 \
# 	--executor-memory 20g \
# 	--driver-memory 20g \
# 	--executor-cores 4 \
# 	--num-executors 4 \
#         -i 20
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib inception failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#2 Total time cost ${time} seconds"

# echo "#3 start test for dllib textclassifier"
# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/news20
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/news20 /tmp/news20
# ls /tmp/news20
# python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/textclassifier/textclassifier.py --on-yarn --max_epoch 3 --model cnn
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib textclassifier failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#3 Total time cost ${time} seconds"


# echo "#4 start test for dllib autograd custom"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/autograd/custom.py --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib autograd custom failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#4 Total time cost ${time} seconds"


# echo "#5 start test for dllib autograd customloss"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/autograd/customloss.py --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib autograd customloss failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#5 Total time cost ${time} seconds"


# echo "#6 start test for dllib nnframes_imageInference"

# if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
#   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
# else
#   wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
#     -P analytics-zoo-models
# fi

# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageInference/ImageInferenceExample.py \
#   -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
#   -f ${HDFS_URI}/kaggle/train_100 --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib nnframes_imageInference failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#6 Total time cost ${time} seconds"


# echo "#7 start test for dllib nnframes_imageTransfer learning"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
#   -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
#   -f ${HDFS_URI}/dogs_cats/samples --nb_epoch 2 --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib nnframes_imageTransfer learning failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#7 Total time cost ${time} seconds"



ls /data
cat /etc/hostname
echo "###################"

# echo "##23 start test for orca data spark_pandas"
# #timer succeed
# start=$(date "+%s")
# #run the example
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/nyc_taxi.csv /data/nyc_taxi.csv
# python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py --deploy-mode 'yarn-client'  -f /data/nyc_taxi.csv
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca data spark_pandas failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#23 Total time cost ${time} seconds"

# echo "#24 start test for pytorch cifar10"
# #timer  succeed
# start=$(date "+%s")
# #run the example
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/cifar10_data /data/cifar10_data
# python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py --cluster_mode 'yarn-client' --epochs 1  --batch_size 256 --data_dir /data/cifar10_data --download False
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca pytorch cifar10 failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#24 Total time cost ${time} seconds"

# echo "#25 start test for pytorch fashion_mnist"
# #timer succeed
# start=$(date "+%s")
# #run the example
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/fashion_mnist/ /data/
# python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py --cluster_mode 'yarn-client'   --epochs 1  --batch_size 256 --download False --data_dir /data/fashion_mnist
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca pytorch fashion_mnist failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#25 Total time cost ${time} seconds"

# echo "#26 start test for pytorch super_resolution"
# #timer succeed
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py  --cluster_mode 'yarn-client' --data_dir /data/dataset 
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca pytorch super_resolution failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#26 Total time cost ${time} seconds"


# echo "#21 start test for orca torchmodel mnist"
# #timer  succeed 
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py  --deploy-mode 'yarn-client' --dir /data/test_mnist --download False
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca torchmodel mnist failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#21 Total time cost ${time} seconds"


# echo "#20 start test for orca torchmodel imagenet"
# #timer  
# start=$(date "+%s")
# #run the example
# # rm -rf /home/imagenet2012
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/imagenet2012 /data/imagenet2012
# python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/imagenet/main.py  /data/imagenet2012 --batch-size 8 --max_epochs 1 --deploy_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca torchmodel imagenet failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#20 Total time cost ${time} seconds"


# echo "#22 start test for orca torchmodel resnet_finetune"
# #timer
# start=$(date "+%s")
# #run the example
# # ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/dogs_cats /data/dogs_cats
# python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/resnet_finetune/resnet_finetune.py ${HDFS_URI}/dogs_cats/samples
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca torchmodel resnet_finetune failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#22 Total time cost ${time} seconds"


#########################  cluster
# echo "##23 start test for orca data spark_pandas"
# #timer succeed
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py --deploy-mode 'yarn-cluster'  -f /data/nyc_taxi.csv
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca data spark_pandas failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#23 Total time cost ${time} seconds"

# echo "#25 start test for pytorch fashion_mnist"
# #timer succeed
# start=$(date "+%s")
# #run the example
# # ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/fashion_mnist/ /data/
# python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py --cluster_mode yarn-cluster   --epochs 1  --batch_size 8 --download False --data_dir /data/fashion_mnist
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca pytorch fashion_mnist failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#25 Total time cost ${time} seconds"


# echo "#21 start test for orca torchmodel mnist"
# #timer  succeed 
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py  --deploy-mode yarn-cluster --dir /data/test_mnist --download False
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca torchmodel mnist failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#21 Total time cost ${time} seconds"


# echo "#20 start test for orca torchmodel imagenet"
# #timer  succeed 
# start=$(date "+%s")
# #run the example
# # rm -rf /home/imagenet2012
# # ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/imagenet2012 /data/imagenet2012 
# python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/imagenet/main.py  /data/imagenet2012 --batch-size 8 --max_epochs 1 --deploy_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca torchmodel imagenet failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#20 Total time cost ${time} seconds"

echo "#24 start test for pytorch cifar10"
#timer  succeed
start=$(date "+%s")
#run the example
${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/cifar10_data /data/cifar10_data
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py --cluster_mode yarn-cluster --epochs 1  --batch_size 8 --data_dir /data/cifar10_data --download False --executor_memory '2g'
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "orca pytorch cifar10 failed"
  #exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#24 Total time cost ${time} seconds"


echo "#22 start test for orca torchmodel resnet_finetune"
#timer
start=$(date "+%s")
#run the example
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/dogs_cats /data/dogs_cats
python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/resnet_finetune/resnet_finetune.py /data/dogs_cats/samples --deploy-mode yarn-cluster
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "orca torchmodel resnet_finetune failed"
  #exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#22 Total time cost ${time} seconds"

echo "#26 start test for pytorch super_resolution"
#timer succeed
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py  --cluster_mode yarn-cluster --data_dir /data/dataset 
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "orca pytorch super_resolution failed"
  #exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#26 Total time cost ${time} seconds"


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
#         --batchSize 128 \
#         --learningRate 0.065 \
#         --weightDecay 0.0002 \
#         --executor-memory 20g \
#         --driver-memory 20g \
#         --executor-cores 4 \
#         --num-executors 4 \
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

# echo "##11 start test for data spark_pandas"
# #timer
# start=$(date "+%s")
# #run the example
# # wget -P /tmp https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
# python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py --deploy-mode 'yarn-client'  -f ${HDFS_URI}/nyc_taxi.csv
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "####################data spark_pandas failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#11 Total time cost ${time} seconds"


# echo "#12 start test for pytorch cifar10"
# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/cifar10_data
# python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py \
#   --cluster_mode 'yarn-client' \
#   --epochs 1 \
#   --batch_size 256
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "####################pytorch cifar10 failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#12 Total time cost ${time} seconds"


# echo "#13 start test for pytorch fashion_mnist"
# #timer
# start=$(date "+%s")
# #run the example
# rm -rf ./data
# rm -rf ./runs
# python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist.py \
#   --cluster_mode 'yarn-client' \
#   --epochs 1 \
#   --batch_size 256
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "####################pytorch fashion_mnist failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#13 Total time cost ${time} seconds"


echo "#14 start test for pytorch super_resolution"
#timer
start=$(date "+%s")
#run the example
# ls /tmp/super_resolution_data
# rm -rf /tmp/super_resolution_data
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py  --cluster_mode 'yarn-client' --data_dir ${HDFS_URI}/tmp/super_resolution_data
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "####################pytorch super_resolution failed"
  #exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#14 Total time cost ${time} seconds"


echo "#15 start test for torchmodel imagenet"
#timer
start=$(date "+%s")
#run the example
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/imagenet_jpeg/ /tmp/imagenet_jpeg/
python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/imagenet/main.py  ${HDFS_URI}/imagenet_jpeg  --max_epochs 1  --batch-size 256  --deploy_mode 'yarn-client'
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "####################torchmodel imagenet failed"
  #exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#15 Total time cost ${time} seconds"


# echo "#16 start test for torchmodel mnist"##
# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/torchmodel_mnist
# python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/mnist/main.py  --dir /tmp/torchmodel_mnist  --epochs 1  --batch-size 256  --deploy-mode 'yarn-client'
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "####################torchmodel mnist failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#16 Total time cost ${time} seconds"


# echo "#17 start test for torchmodel resnet_finetune"
# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/dogs_cats/samples
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/dogs_cats/ /tmp/dogs_cats/
# echo "#########################samples"
# ls /tmp/dogs_cats/samples
# python ${BIGDL_ROOT}/python/orca/example/torchmodel/train/resnet_finetune/resnet_finetune.py ${HDFS_URI}/dogs_cats/samples
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "####################torchmodel resnet_finetune failed"
#   #exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#17 Total time cost ${time} seconds"

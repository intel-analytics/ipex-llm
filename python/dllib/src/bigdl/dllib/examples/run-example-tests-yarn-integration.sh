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

# echo "#4 start test for orca bigdl transformer"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/transformer.py \
#   --cluster_mode yarn_cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca transformer failed"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#4 Total time cost ${time} seconds"


echo "#5 start test for orca bigdl imageInference"
#timer
start=$(date "+%s")
if [ -f models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
  echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
  wget -nv $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P models
fi
#run the example
#python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/imageInference/imageInference.py \
python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageInference/ImageInferenceExample.py
  -m models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/kaggle/train_100 --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca imageInference failed"
#   exit $exit_status
# fi
now=$(date "+%s")
time=$((now - start))
echo "#5 Total time cost ${time} seconds"

# echo "#6 start test for orca pytorch_estimator"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/horovod/pytorch_estimator.py --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca pytorch_estimator failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#6 Total time cost ${time} seconds"

# echo "#7 start test for orca simple_pytorch"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/horovod/simple_horovod_pytorch.py --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca simple_pytorch failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#7 Total time cost ${time} seconds"

# echo "#7 start test for orca mxnet"
# #timer
# start=$(date "+%s")

# if [ -f ${BIGDL_ROOT}/data/mnist.zip ]
# then
#     echo "mnist.zip already exists"
# else
#     wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P ${BIGDL_ROOT}/data
# fi
# unzip -q ${BIGDL_ROOT}/data/mnist.zip -d ${BIGDL_ROOT}/data

# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/mxnet/lenet_mnist.py --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca mxnet failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#7 Total time cost ${time} seconds"

# echo "#8 start test for orca openvino"
# #timer
# start=$(date "+%s")
# if [ -f models/faster_rcnn_resnet101_coco.xml ]; then
#   echo "models/faster_rcnn_resnet101_coco already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.xml \
#     -P models
#   wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.bin \
#     -P models
# fi
# if [ -d tmp/data/object-detection-coco ]; then
#   echo "tmp/data/object-detection-coco already exists"
# else
#   wget -nv $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P tmp/data
#   unzip -q tmp/data/object-detection-coco.zip -d tmp/data
# fi
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/openvino/predict.py \
#   --image tmp/data/object-detection-coco \
#   --model models/faster_rcnn_resnet101_coco.xml \
#   --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca openvino failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#8 Total time cost ${time} seconds"

# echo "#prepare dataset for ray_on_spark"
# wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz
# wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz
# wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz
# wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz
# zip MNIST_data.zip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

# echo "#9 start test for orca ros async"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/async_parameter_server.py \
#   --iterations 5 \
#   --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca ros async failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#9 Total time cost ${time} seconds"

# echo "#10 start test for orca ros sync"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/sync_parameter_server.py \
#   --iterations 5 \
#   --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca ros sync failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#10 Total time cost ${time} seconds"

# echo "#11 start test for orca rllib"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/rllib/multiagent_two_trainers.py \
#   --iterations 5 \
#   --cluster_mode yarn-cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca ros rllib failed"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#11 Total time cost ${time} seconds"

# echo "#12 start test for orca rl_pong"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/rl_pong/rl_pong.py \
#   --iterations 5 \
#   --cluster_mode yarn-cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca ros rl_pong failed"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#12 Total time cost ${time} seconds"

# echo "prepare dataset for tfpark keras"
# rm -f /tmp/mnist/*
# wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P /tmp/mnist
# wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P /tmp/mnist
# wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P /tmp/mnist
# wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P /tmp/mnist

# echo "#13 start test for orca tfpark keras_dataset"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/tfpark/keras/keras_dataset.py \
#   --max_epoch 5 \
#   --cluster_mode yarn-cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca ros rl_pong failed"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#13 Total time cost ${time} seconds"

# echo "#14 start test for orca tfpark keras_dataset"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/tfpark/keras/keras_ndarray.py \
#   --max_epoch 5 \
#   --cluster_mode yarn-cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca ros rl_pong failed"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#14 Total time cost ${time} seconds"

# echo "#15 start test for orca tfpark gan"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/tfpark/gan/gan_train_and_evaluate.py \
#   --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca tfpark gan failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#15 Total time cost ${time} seconds"

# echo "#16 start test for orca tfpark estimator_dataset"
# #timer 
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/tfpark/estimator/estimator_dataset.py \
#   --cluster_mode yarn-cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca tfpark estimator_dataset"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#16 Total time cost ${time} seconds"

# echo "#17 start test for orca tfpark estimator_inception"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/tfpark/estimator/estimator_inception.py \
#   --image-path ${HDFS_URI}/dogs_cats \
#   --num-classes 2 \
#   --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca tfpark estimator_inception failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#17 Total time cost ${time} seconds"

# echo "#18 start test for orca tfpark optimizer train"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/tfpark/tf_optimizer/train.py \
#   --max_epoch 1 \
#   --data_num 1000 \
#   --cluster_mode yarn-cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca tfpark optimizer train failed"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#18 Total time cost ${time} seconds"

# echo "#19 start test for orca tfpark optimizer evaluate"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/tfpark/tf_optimizer/evaluate.py \
#   --data_num 1000 \
#   --cluster_mode yarn-cluster
# # exit_status=$?
# # if [ $exit_status -ne 0 ]; then
# #   clear_up
# #   echo "orca tfpark optimizer evaluate failed"
# #   exit $exit_status
# # fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#19 Total time cost ${time} seconds"

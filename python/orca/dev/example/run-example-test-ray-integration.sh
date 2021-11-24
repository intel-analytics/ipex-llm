#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling BigDL"
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

set -e

echo "#start orca ray example tests"
# echo "#1 Start autoestimator example"
# start=$(date "+%s")
# python ${BIGDL_ROOT}/python/orca/example/automl/autoestimator/autoestimator_pytorch.py --trials 5 --epochs 2 --cluster_mode yarn
# now=$(date "+%s")
# time1=$((now-start))

# echo "#2 Start autoxgboost example"
# if [ -f ${BIGDL_ROOT}/data/airline_14col.data ]
# then
#     echo "airline_14col.data already exists"
# else
#     wget -nv $FTP_URI/analytics-zoo-data/airline_14col.data -P ${BIGDL_ROOT}/data/
# fi

# start=$(date "+%s")
# python ${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostClassifier.py -p ${BIGDL_ROOT}/data/airline_14col.data --cluster_mode yarn
# now=$(date "+%s")
# time2=$((now-start))

# echo "#3 Start autoxgboost example"
# if [ -f ${BIGDL_ROOT}/data/incd.csv ]
# then
#     echo "incd.csv already exists"
# else
#     wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${BIGDL_ROOT}/data/
# fi

# start=$(date "+%s")
# python ${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostRegressor.py -p ${BIGDL_ROOT}/data/incd.csv --cluster_mode yarn
# now=$(date "+%s")
# time3=$((now-start))

# echo "#4 start test for orca bigdl transformer"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/transformer.py \
#   --cluster_mode yarn_client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca transformer failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#4 Total time cost ${time} seconds"

# echo "#5 start test for orca bigdl imageInference"
# #timer
# start=$(date "+%s")
# if [ -f models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
#   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
# else
#   wget -nv $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
#     -P models
# fi
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/imageInference/imageInference.py \
#   -m models/bigdl_inception-v1_imagenet_0.4.0.model \
#   -f ${HDFS_URI}/kaggle/train_100 --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca imageInference failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#5 Total time cost ${time} seconds"

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

# echo "#8 start test for orca mxnet"
# #timer
# start=$(date "+%s")

# # if [ -f ${BIGDL_ROOT}/data/mnist.zip ]
# # then
# #     echo "mnist.zip already exists"
# # else
# #     wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P ${BIGDL_ROOT}/data
# # fi
# # unzip -q ${BIGDL_ROOT}/data/mnist.zip -d ${BIGDL_ROOT}/data

# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/mxnet/lenet_mnist.py #--cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca mxnet failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#8 Total time cost ${time} seconds"

echo "#prepare dataset for ray_on_spark"
wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz
wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz
wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz
wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz
zip ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/MNIST_data.zip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

echo "#9 start test for orca ros async"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/sync_parameter_server.py --iterations 20 --num_workers 2 --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca ros async failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#9 Total time cost ${time} seconds"

echo "#10 start test for orca ros sync"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/async_parameter_server.py --iterations 20 --num_workers 2 --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca ros sync failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#10 Total time cost ${time} seconds"

echo "#11 start test for orca rllib"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/rllib/multiagent_two_trainers.py \
  --iterations 5 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca ros rllib failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#11 Total time cost ${time} seconds"

echo "#12 start test for orca rl_pong"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/rl_pong/rl_pong.py \
  --iterations 5 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca ros rl_pong failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#12 Total time cost ${time} seconds"

echo "#13 start test for orca tfpark keras_dataset"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/tfpark/keras/keras_dataset.py \
  --data_path ${HDFS_URI}/mnist \
  --max_epoch 5 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca ros rl_pong failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#13 Total time cost ${time} seconds"

echo "#14 start test for orca tfpark keras_dataset"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/tfpark/keras/keras_ndarray.py \
  --max_epoch 5 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca ros rl_pong failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#14 Total time cost ${time} seconds"

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

echo "#16 start test for orca tfpark estimator_dataset"
#timer 
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/tfpark/estimator/estimator_dataset.py \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tfpark estimator_dataset"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#16 Total time cost ${time} seconds"

echo "#17 start test for orca tfpark estimator_inception"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/tfpark/estimator/estimator_inception.py \
  --image-path ${HDFS_URI}/dogs_cats \
  --num-classes 2 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tfpark estimator_inception failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#17 Total time cost ${time} seconds"

echo "#18 start test for orca tfpark optimizer train"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/tfpark/tf_optimizer/train.py \
  --max_epoch 1 \
  --data_num 1000 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tfpark optimizer train failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#18 Total time cost ${time} seconds"

echo "#19 start test for orca tfpark optimizer evaluate"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/tfpark/tf_optimizer/evaluate.py \
  --data_num 1000 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tfpark optimizer evaluate failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#19 Total time cost ${time} seconds"

echo "Ray example tests finished"

echo "#1 auto-estimator-pytorch time used:$time1 seconds"
echo "#2 auto-xgboost-classifier time used:$time2 seconds"
echo "#3 auto-xgboost-regressor time used:$time3 seconds"
echo "#4 bigdl transformer time used:$time4 seconds"
echo "#5 bigdl imageInference time used:$time5 seconds"
echo "#6 horovod pytorch_estimator time used:$time6 seconds"
#echo "#7 orca multiagent_two_trainers time used:$time7 seconds"
#echo "#8 mxnet_lenet time used:$time8 seconds"
echo "#9 paramerter_server async time used:$time9 seconds"
echo "#10 paramerter_server sync example time used:$time10 seconds"
echo "#11 paramerter_server rllib example time used:$time11 seconds"
echo "#12 paramerter_server rl_pong example time used:$time12 seconds"
echo "#13 tfaprk keras_dataset example time used:$time13 seconds"
echo "#14 tfaprk keras_ndarray example time used:$time14 seconds"
#echo "#15 tfaprk gan_train_and_evaluate example time used:$time15 seconds"
echo "#16 tfaprk estimator_dataset example time used:$time16 seconds"
echo "#17 tfaprk estimator_inception example time used:$time17 seconds"
echo "#18 tfaprk opt_train example time used:$time18 seconds"
echo "#19 tfaprk opt_evaluate example time used:$time19 seconds"


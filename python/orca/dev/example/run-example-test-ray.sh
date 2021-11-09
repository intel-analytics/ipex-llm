#!/bin/bash

set -e

ray stop -f

clear_up () {
    echo "Clearing up environment. Uninstalling BigDL"
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

execute_ray_test(){
    echo "start example $1"
    start=$(date "+%s")
    python $2
    exit_status=$?
    if [ $exit_status -ne 0 ];
    then
        clear_up
        echo "$1 failed"
        exit $exit_status
    fi
    now=$(date "+%s")
    return $((now-start))
}

echo "#start orca ray example tests"
echo "#1 Start rl_pong example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/rl_pong/rl_pong.py --iterations 10
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start multiagent example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/rllib/multiagent_two_trainers.py --iterations 5
now=$(date "+%s")
time2=$((now-start))

echo "#3 Start async_parameter example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/async_parameter_server.py --iterations 10
now=$(date "+%s")
time3=$((now-start))

echo "#4 Start sync_parameter example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/sync_parameter_server.py --iterations 10
now=$(date "+%s")
time4=$((now-start))

echo "#5 Start mxnet lenet example"
start=$(date "+%s")

# get_mnist_iterator in MXNet requires the data to be placed in the `data` folder of the running directory.
# The running directory of integration test is ${ANALYTICS_ZOO_ROOT}.
if [ -f tmp/data/mnist.zip ]
then
    echo "mnist.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P tmp/data
fi
unzip -q tmp/data/mnist.zip -d tmp/data

python ${BIGDL_ROOT}/python/orca/example/learn/mxnet/lenet_mnist.py -e 1 -b 256
now=$(date "+%s")
time5=$((now-start))

echo "#6 Start fashion_mnist example with Tensorboard visualization"
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

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py --backend torch_distributed
now=$(date "+%s")
time6=$((now-start))


echo "#7 start example for orca super-resolution"
start=$(date "+%s")

if [ ! -f BSDS300-images.tgz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/BSDS300-images.tgz
fi
if [ ! -d dataset/BSDS300/images ]; then
  mkdir dataset
  tar -xzf BSDS300-images.tgz -C dataset
fi

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py --backend torch_distributed

now=$(date "+%s")
time7=$((now-start))


echo "#8 start example for orca cifar10"
start=$(date "+%s")

if [ -d ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/data ]; then
  echo "Cifar10 already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10
  unzip ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.zip
fi

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py --backend torch_distributed

now=$(date "+%s")
time8=$((now-start))

execute_ray_test auto-estimator-pytorch "${BIGDL_ROOT}/python/orca/example/automl/autoestimator/autoestimator_pytorch.py --trials 5 --epochs 2"
time9=$?

if [ -f ${BIGDL_ROOT}/data/airline_14col.data ]
then
    echo "airline_14col.data already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/airline_14col.data -P ${BIGDL_ROOT}/data/
fi

execute_ray_test auto-xgboost-classifier "${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostClassifier.py -p ${BIGDL_ROOT}/data/airline_14col.data"
time10=$?

if [ -f ${BIGDL_ROOT}/data/incd.csv ]
then
    echo "incd.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${BIGDL_ROOT}/data/
fi

execute_ray_test auto-xgboost-regressor "${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostRegressor.py -p ${BIGDL_ROOT}/data/incd.csv"
time11=$?


echo "Ray example tests finished"
echo "#1 orca rl_pong time used:$time1 seconds"
echo "#2 orca async_parameter_server time used:$time2 seconds"
echo "#3 orca sync_parameter_server time used:$time3 seconds"
echo "#4 orca multiagent_two_trainers time used:$time4 seconds"
echo "#5 mxnet_lenet time used:$time5 seconds"
echo "#6 fashion-mnist time used:$time6 seconds"
echo "#7 orca super-resolution example time used:$time7 seconds"
echo "#8 orca cifar10 example time used:$time8 seconds"
echo "#9 auto-estimator-pytorch time used:$time1 seconds"
echo "#10 auto-xgboost-classifier time used:$time2 seconds"
echo "#11 auto-xgboost-regressor time used:$time3 seconds"

#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH
export BIGDL_JARS=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

set -e

ray stop -f

echo "#start orca ray example tests"
echo "#1 Start rl_pong example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/rl_pong/rl_pong.py --iterations 10
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start multiagent example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/rllib/multiagent_two_trainers.py --iterations 5
now=$(date "+%s")
time2=$((now-start))

echo "#3 Start async_parameter example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/parameter_server/async_parameter_server.py --iterations 10
now=$(date "+%s")
time3=$((now-start))

echo "#4 Start sync_parameter example"
#start=$(date "+%s")
#python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/parameter_server/sync_parameter_server.py --iterations 10
#now=$(date "+%s")
#time4=$((now-start))

echo "#5 Start mxnet lenet example"
start=$(date "+%s")

# get_mnist_iterator in MXNet requires the data to be placed in the `data` folder of the running directory.
# The running directory of integration test is ${ANALYTICS_ZOO_ROOT}.
if [ -f ${ANALYTICS_ZOO_ROOT}/data/mnist.zip ]
then
    echo "mnist.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P ${ANALYTICS_ZOO_ROOT}/data
fi
unzip -q ${ANALYTICS_ZOO_ROOT}/data/mnist.zip -d ${ANALYTICS_ZOO_ROOT}/data

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/mxnet/lenet_mnist.py -e 1 -b 256
now=$(date "+%s")
time5=$((now-start))

echo "#6 Start fashion_mnist example with Tensorboard visualization"
start=$(date "+%s")

if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/data ]
then
    echo "fashion-mnist already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/fashion-mnist.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/
    unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion-mnist.zip
fi

sed "s/epochs=5/epochs=1/g;s/batch_size=4/batch_size=256/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py --backend torch_distributed
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

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/super_resolution/super_resolution.py --backend torch_distributed

now=$(date "+%s")
time7=$((now-start))


echo "#8 start example for orca cifar10"
start=$(date "+%s")

if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/data ]; then
  echo "Cifar10 already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10
  unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.zip
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.py --backend torch_distributed

now=$(date "+%s")
time8=$((now-start))

echo "#9 start example for orca auto-xgboost-classifier"
start=$(date "+%s")

if [ -f ${ANALYTICS_ZOO_ROOT}/data/airline_14col.data ]
then
    echo "airline_14col.data already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/airline_14col.data -P ${ANALYTICS_ZOO_ROOT}/data/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/AutoXGBoostClassifier.py \
 -p ${ANALYTICS_ZOO_ROOT}/data/airline_14col.data

now=$(date "+%s")
time9=$((now-start))


echo "#10 start example for orca auto-xgboost-regressor"
start=$(date "+%s")

if [ -f ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/incd.csv ]
then
    echo "incd.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/AutoXGBoostRegressor.py \
 -p ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/incd.csv

now=$(date "+%s")
time10=$((now-start))


echo "#11 start example for orca autoestimator-pytorch"
start=$(date "+%s")

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoestimator/autoestimator_pytorch.py \
    --trials 5 --epochs 2

now=$(date "+%s")
time11=$((now-start))


echo "#12 start example for chronos autolstm_nyc_taxi"
start=$(date "+%s")

if [ -f ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ]
then
    echo "nyc_taxi.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/autolstm_nyc_taxi.py

now=$(date "+%s")
time12=$((now-start))

echo "#13 start example for chronos autoprophet_nyc_taxi"
start=$(date "+%s")

if [ -f ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/nyc_taxi.csv ]
then
    echo "nyc_taxi.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/autoprophet_nyc_taxi.py \
    --datadir ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/nyc_taxi.csv \
    --n_sampling 2

now=$(date "+%s")
time13=$((now-start))

echo "#14 start example for chronos simulator-dpgansimulator-wwt"
start=$(date "+%s")

if [ -f ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/data_train_small.npz ]
then
    echo "data_train_small.npz already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/apps/doppelGANger_data/data_train_small.npz -P \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/dpgansimulator_wwt.py \
    --datadir ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/data_train_small.npz\
    --epoch 1 --plot_figures False

now=$(date "+%s")
time14=$((now-start))

echo "#15 start example for chronos distributed_training_network_traffic"
start=$(date "+%s")

if [ -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]
then
    echo "network_traffic_data.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
    mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/distributed/distributed_training_network_traffic.py

now=$(date "+%s")
time15=$((now-start))

echo "#16 start example for chronos onnx_autotsestimator_nyc_taxi"
start=$(date "+%s")

if [ ! -f ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv ]
then
    wget $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
    mv ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv
else
    echo "nyc_taxi_data.csv exists."
fi

# When the thread of onnxruntime is None, "pthread_setaffinity_np failed" may appear.
sed -i '/onnx/d' ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/onnx/onnx_autotsestimator_nyc_taxi.py

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/onnx/onnx_autotsestimator_nyc_taxi.py

now=$(date "+%s")
time16=$((now-start))

echo "#17 start example for chronos onnx_autotsestimator_nyc_taxi"
start=$(date "+%s")

if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]
then
    wget $FTP_URI/analytics-zoo-data/network_traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
    mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/nyc_taxi_data.csv
else
    echo "network_traffic_data.csv exists."
fi

# When the thread of onnxruntime is None, "pthread_setaffinity_np failed" may appear.
sed -i '/onnx/d' ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/onnx/onnx_forecaster_network_traffic.py

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/onnx/onnx_forecaster_network_traffic.py

now=$(date "+%s")
time17=$((now-start))

echo "Ray example tests finished"
echo "#1 orca rl_pong time used:$time1 seconds"
echo "#2 orca async_parameter_server time used:$time2 seconds"
echo "#3 orca sync_parameter_server time used:$time3 seconds"
echo "#4 orca multiagent_two_trainers time used:$time4 seconds"
echo "#5 mxnet_lenet time used:$time5 seconds"
echo "#6 fashion-mnist time used:$time6 seconds"
echo "#7 orca super-resolution example time used:$time7 seconds"
echo "#8 orca cifar10 example time used:$time8 seconds"
echo "#9 orca auto-xgboost-classifier time used:$time9 seconds"
echo "#10 orca auto-xgboost-regressor time used:$time10 seconds"
echo "#11 orca autoestimator-pytorch time used:$time11 seconds"
echo "#12 chronos autolstm_nyc_taxi time used:$time12 seconds"
echo "#13 chronos autoprophet_nyc_taxi time used:$time13 seconds"
echo "#14 chronos simulator-dpgansimulator-wwt time used:$time14 seconds"
echo "#15 chronos distributed_training_network_traffic time used:$time15 seconds"
echo "#16 chronos onnx_autotsestimator_nyc_taxi time used:$time16 seconds"
echo "#17 chronos onnx_forecaster_network_traffic time used:$time17 seconds"

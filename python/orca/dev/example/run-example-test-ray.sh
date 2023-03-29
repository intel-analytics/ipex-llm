#!/usr/bin/env bash
set -e
set -x

clear_up () {
    echo "Clearing up environment. Uninstalling BigDL"
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

python_version=$(python --version | awk '{print$2}')

echo "#start orca ray example tests"
echo "#1 Start autoestimator example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/automl/autoestimator/autoestimator_pytorch.py --trials 5 --epochs 2
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start autoxgboost example"
if [ -f ${BIGDL_ROOT}/data/airline_14col.data ]
then
    echo "airline_14col.data already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/airline_14col.data -P ${BIGDL_ROOT}/data/
fi

start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostClassifier.py -p ${BIGDL_ROOT}/data/airline_14col.data
now=$(date "+%s")
time2=$((now-start))

echo "#3 Start autoxgboost example"
if [ -f ${BIGDL_ROOT}/data/incd.csv ]
then
    echo "incd.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${BIGDL_ROOT}/data/
fi

start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostRegressor.py -p ${BIGDL_ROOT}/data/incd.csv
now=$(date "+%s")
time3=$((now-start))

set -e
ray stop -f

if [ $python_version == 3.7.10 ];then
# rllib test does not support numpy 1.24
# parameter server test requires tensorflow 1
# mxnet test does not support numpy 1.24
echo "#4 Start multi_agent example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/rllib/multi_agent_two_trainers.py --iterations 5
now=$(date "+%s")
time4=$((now-start))

echo "#5 Start async_parameter example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/async_parameter_server.py --iterations 10
now=$(date "+%s")
time5=$((now-start))

echo "#6 Start sync_parameter example"
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/ray_on_spark/parameter_server/sync_parameter_server.py --iterations 10
now=$(date "+%s")
time6=$((now-start))

echo "#7 Start mxnet lenet example"
start=$(date "+%s")

# get_mnist_iterator in MXNet requires the data to be placed in the `data` folder of the running directory.
# The running directory of integration test is ${ANALYTICS_ZOO_ROOT}.
if [ -f data/mnist.zip ]
then
    echo "mnist.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P data
fi
unzip -q data/mnist.zip -d data

python ${BIGDL_ROOT}/python/orca/example/learn/mxnet/lenet_mnist.py -e 1 -b 256
now=$(date "+%s")
time7=$((now-start))
fi

echo "#8 Start fashion_mnist example with Tensorboard visualization"
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

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py --backend ray --batch_size=256
now=$(date "+%s")
time8=$((now-start))


echo "#9 start example for orca super-resolution"
start=$(date "+%s")

if [ ! -f BSDS300-images.tgz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/BSDS300-images.tgz
fi
if [ ! -d dataset/BSDS300/images ]; then
  mkdir dataset
  tar -xzf BSDS300-images.tgz -C dataset
fi

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/super_resolution/super_resolution.py --backend ray

now=$(date "+%s")
time9=$((now-start))


echo "#10 start example for orca cifar10"
start=$(date "+%s")

if [ -d ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/data ]; then
  echo "Cifar10 already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10
  unzip ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.zip
fi

python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/cifar10/cifar10.py --backend ray --batch_size=256

now=$(date "+%s")
time10=$((now-start))


echo "#11 Start autoxgboost example"
if [ -f ${BIGDL_ROOT}/data/incd.csv ]
then
    echo "incd.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${BIGDL_ROOT}/data/
fi

start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostRegressor_spark_df.py -p ${BIGDL_ROOT}/data/incd.csv
now=$(date "+%s")
time11=$((now-start))


echo "#12 Start ray dataset xgboost example"
if [ -f ${BIGDL_ROOT}/data/incd.csv ]
then
    echo "incd.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${BIGDL_ROOT}/data/
fi

start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/data/ray-dataset-xgboost.py -p ${BIGDL_ROOT}/data/incd.csv
now=$(date "+%s")
time12=$((now-start))

echo "#13 start example for orca brainMRI"
if [ -f ${BIGDL_ROOT}/python/orca/example/learn/pytorch/brainMRI/kaggle_3m ]
then
    echo "kaggle_3m already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/kaggle_3m.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/brainMRI
    unzip ${BIGDL_ROOT}/python/orca/example/learn/pytorch/brainMRI/kaggle_3m.zip
fi

start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/brainMRI/brainMRI.py --epochs=1
export PYTHONPATH=${BIGDL_ROOT}/python/orca/example/learn/pytorch/brainMRI:$PYTHONPATH
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/brainMRI/brainMRI.py --backend=spark --epochs=1
now=$(date "+%s")
time13=$((now-start))

echo "#14 start example for orca resnet50 inference"
if [ -f ${BIGDL_ROOT}/python/orca/example/learn/pytorch/resnet50/imagenet-small ]
then
    echo "imagenet-small already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/imagenet-small.zip -P ${BIGDL_ROOT}/python/orca/example/learn/pytorch/resnet50
    unzip -d ${BIGDL_ROOT}/python/orca/example/learn/pytorch/resnet50 ${BIGDL_ROOT}/python/orca/example/learn/pytorch/resnet50/imagenet-small.zip
fi

start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/learn/pytorch/resnet50/inference.py ${BIGDL_ROOT}/python/orca/example/learn/pytorch/resnet50/imagenet-small -w 2 --cores 8 --workers_per_node 2 --steps 10 -j 0
now=$(date "+%s")
time14=$((now-start))

echo "#15 start example test for orca data"
if [ -f tmp/data/NAB/nyc_taxi/nyc_taxi.csv ]; then
  echo "tmp/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
    -P tmp/data/NAB/nyc_taxi/
fi
#timer
start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py \
  -f tmp/data/NAB/nyc_taxi/nyc_taxi.csv

now=$(date "+%s")
time15=$((now - start))

echo "#16 start example for mmcv faster-rcnn training"
if [ -f ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/kitti_tiny ]
then
    echo "kitti_tiny already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mmcv/kitti_tiny.zip -P ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn
    unzip -d ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/kitti_tiny.zip
fi

if [ -f ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth ]
then
    echo "faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mmcv/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth -P ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn
fi

if [ -f ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/mmdetection-master ]
then
    echo "mmdetection-master already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mmcv/mmdetection-master.zip -P ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn
    unzip -d ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/mmdetection-master.zip
fi

start=$(date "+%s")
python ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/train.py --dataset ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/kitti_tiny/ --config ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/mmdetection-master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py --load_from ${BIGDL_ROOT}/python/orca/example/learn/mmcv/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
now=$(date "+%s")
time16=$((now-start))

echo "Ray example tests finished"

echo "#1 auto-estimator-pytorch time used:$time1 seconds"
echo "#2 auto-xgboost-classifier time used:$time2 seconds"
echo "#3 auto-xgboost-regressor time used:$time3 seconds"
echo "#4 orca async_parameter_server time used:$time4 seconds"
echo "#5 orca sync_parameter_server time used:$time5 seconds"
echo "#6 orca multi_agent_two_trainers time used:$time6 seconds"
echo "#7 mxnet_lenet time used:$time7 seconds"
echo "#8 fashion-mnist time used:$time8 seconds"
echo "#9 orca super-resolution example time used:$time9 seconds"
echo "#10 orca cifar10 example time used:$time10 seconds"
echo "#11 auto-xgboost-regressor-spark-df example time used:$time11 seconds"
echo "#12 ray-dataset-xgboost example time used:$time12 seconds"
echo "#13 orca brainMRI example time used:$time13 seconds"
echo "#14 orca resnet50 inference example time used:$time14 seconds"
echo "#15 orca data time used:$time15 seconds"
echo "#16 mmcv faster_rcnn training example time used:$time16 seconds"

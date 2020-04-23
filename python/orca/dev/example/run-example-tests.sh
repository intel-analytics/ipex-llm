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

set -e

echo "#1 start example test for textclassification"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists" 
else
    wget -nv $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi 
if [ -f analytics-zoo-data/data/20news-18828.tar.gz ]
then 
    echo "analytics-zoo-data/data/20news-18828.tar.gz already exists" 
else
    wget -nv $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P analytics-zoo-data/data
    tar zxf analytics-zoo-data/data/20news-18828.tar.gz -C analytics-zoo-data/data/
fi

${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --nb_epoch 1 \
    --data_path analytics-zoo-data/data/20news-18828 \
    --embedding_path analytics-zoo-data/data/glove.6B

now=$(date "+%s")
time1=$((now-start))

echo "#2 start example test for autograd"
#timer
start=$(date "+%s")
echo "#2.1 start example test for custom layer"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/custom.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/custom.py \
    --nb_epoch 2

echo "#2.2 start example test for customloss"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/customloss.py

now=$(date "+%s")
time2=$((now-start))

echo "#3 start example test for image-classification"
#timer
start=$(date "+%s")

echo "check if model directory exists"
if [ ! -d analytics-zoo-models ]
then
    mkdir analytics-zoo-models
fi

if [ -f analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0.model ]
then
    echo "analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0.model already exists"
else
    wget -nv $FTP_URI/analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model \
    -P analytics-zoo-models
fi

${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/imageclassification/predict.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/imageclassification/predict.py \
    -f ${HDFS_URI}/kaggle/train_100 \
    --model analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0.model \
    --topN 5
now=$(date "+%s")
time3=$((now-start))

echo "#4 start example test for object-detection"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model ]
then
    echo "analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model already exists"
else
    wget -nv $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model \
    -P analytics-zoo-models
fi

${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/objectdetection/predict.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/objectdetection/predict.py \
    analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model ${HDFS_URI}/kaggle/train_100 /tmp
now=$(date "+%s")
time4=$((now-start))

echo "#5 start example test for nnframes"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]
then
   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
   wget -nv $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi
 if [ -f analytics-zoo-data/data/dogs-vs-cats/train.zip ]
then
   echo "analytics-zoo-data/data/dogs-vs-cats/train.zip already exists."
else
   # echo "Downloading dogs and cats images"
   wget -nv  $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip\
    -P analytics-zoo-data/data/dogs-vs-cats
   unzip -q analytics-zoo-data/data/dogs-vs-cats/train.zip -d analytics-zoo-data/data/dogs-vs-cats
   mkdir -p analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/cat.71* analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/dog.71* analytics-zoo-data/data/dogs-vs-cats/samples

   mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/cats
   mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/dogs
   cp analytics-zoo-data/data/dogs-vs-cats/train/cat.71* analytics-zoo-data/data/dogs-vs-cats/demo/cats
   cp analytics-zoo-data/data/dogs-vs-cats/train/dog.71* analytics-zoo-data/data/dogs-vs-cats/demo/dogs
   # echo "Finished downloading images"
fi
    
echo "start example test for nnframes finetune"
${SPARK_HOME}/bin/spark-submit \
   --master local[2] \
   --driver-memory 10g \
   --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/image_finetuning_example.py \
   --jars ${ANALYTICS_ZOO_JAR} \
   --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
   --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/image_finetuning_example.py \
   -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
   -f analytics-zoo-data/data/dogs-vs-cats/samples
   
echo "start example test for nnframes imageInference"
${SPARK_HOME}/bin/spark-submit \
   --master local[1] \
   --driver-memory 3g \
   --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/ImageInferenceExample.py \
   --jars ${ANALYTICS_ZOO_JAR} \
   --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
   --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/ImageInferenceExample.py \
   -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
   -f ${HDFS_URI}/kaggle/train_100

echo "start example test for nnframes imageTransferLearning"
${SPARK_HOME}/bin/spark-submit \
   --master local[1] \
   --driver-memory 5g \
   --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
   --jars ${ANALYTICS_ZOO_JAR} \
   --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
   --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py\
   -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
   -f analytics-zoo-data/data/dogs-vs-cats/samples


now=$(date "+%s")
time5=$((now-start))

echo "#6 start example test for tensorflow"
#timer
start=$(date "+%s")
echo "start example test for tensorflow tfnet"
if [ -f analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz ]
then
   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
   wget -nv $FTP_URI/analytics-zoo-models/tensorflow/ssd_mobilenet_v1_coco_2017_11_17.tar.gz \
    -P analytics-zoo-models
   tar zxf analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz -C analytics-zoo-models/
fi
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfnet/predict.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfnet/predict.py \
    --image ${HDFS_URI}/kaggle/train_100 \
    --model analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb

echo "start example test for tfpark"
if [ ! -d analytics-zoo-tensorflow-models ]
then
    mkdir analytics-zoo-tensorflow-models
    mkdir -p analytics-zoo-tensorflow-models/mnist
    mkdir -p analytics-zoo-tensorflow-models/az_lenet
    mkdir -p analytics-zoo-tensorflow-models/lenet
fi

sed "s%/tmp%analytics-zoo-tensorflow-models%g;s%models/slim%slim%g"
if [ -d analytics-zoo-tensorflow-models/slim ]
then
    echo "analytics-zoo-tensorflow-models/slim already exists."
else
    echo "Downloading research/slim"
   
   wget -nv $FTP_URI/analytics-zoo-tensorflow-models/models/research/slim.tar.gz -P analytics-zoo-tensorflow-models
   tar -zxvf analytics-zoo-tensorflow-models/slim.tar.gz -C analytics-zoo-tensorflow-models
   
   echo "Finished downloading research/slim"
   export PYTHONPATH=`pwd`/analytics-zoo-tensorflow-models/slim:$PYTHONPATH
 fi

echo "start example test for TFPark tf_optimizer train 1"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train.py 1 1000


echo "start example test for TFPark tf_optimizer evaluate 2"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate.py 1000


echo "start example test for TFPark keras keras_dataset 3"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/keras/keras_dataset.py 1


echo "start example test for TFPark keras keras_ndarray 4"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/keras/keras_ndarray.py 1


echo "start example test for TFPark estimator estimator_dataset 5"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/estimator/estimator_dataset.py


echo "start example test for TFPark estimator estimator_inception 6"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/estimator/estimator_inception.py \
        --image-path analytics-zoo-data/data/dogs-vs-cats/demo --num-classes 2

echo "start example test for TFPark gan 7"

sed "s/MaxIteration(1000)/MaxIteration(5)/g;s/range(20)/range(2)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/gan/gan_train_and_evaluate.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/gan/gan_train_tmp.py

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/gan/gan_train_tmp.py


echo "start example test for TFPark inceptionv1 training 8"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
   --master local[4] \
   --driver-memory 10g \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/inception/inception.py \
   --maxIteration 20 \
   -b 8 \
   -f ${HDFS_URI}/imagenet-mini

if [ -f analytics-zoo-models/resnet_50_saved_model.zip ]
then
   echo "analytics-zoo-models/resnet_50_saved_model.zip already exists."
else
   wget -nv $FTP_URI/analytics-zoo-models/tensorflow/resnet_50_saved_model.zip \
    -P analytics-zoo-models
   unzip analytics-zoo-models/resnet_50_saved_model.zip -d analytics-zoo-models/resnet_50_saved_model
fi

echo "start example test for TFPark freeze saved model 9"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
   --master local[4] \
   --driver-memory 10g \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/freeze_saved_model/freeze.py \
        --saved_model_path analytics-zoo-models/resnet_50_saved_model \
        --output_path analytics-zoo-models/resnet_50_tfnet

now=$(date "+%s")
time6=$((now-start))

echo "#7 start example test for anomalydetection"
if [ -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv ]
then
    echo "analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
    -P analytics-zoo-data/data/NAB/nyc_taxi/
fi
#timer
start=$(date "+%s")
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection.py \
    --nb_epoch 1 \
    --input_dir analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv
now=$(date "+%s")
time7=$((now-start))

echo "#8 start example test for qaranker"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi
if [ -f analytics-zoo-data/data/WikiQAProcessed.zip ]
then
    echo "analytics-zoo-data/data/WikiQAProcessed.zip already exists"
else
    echo "downloading WikiQAProcessed.zip"
    wget -nv $FTP_URI/analytics-zoo-data/WikiQAProcessed.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/WikiQAProcessed.zip -d analytics-zoo-data/data/
fi

${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 3g \
    --executor-memory 3g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/qaranker/qa_ranker.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/qaranker/qa_ranker.py \
    --nb_epoch 2 \
    --data_path analytics-zoo-data/data/WikiQAProcessed \
    --embedding_file analytics-zoo-data/data/glove.6B/glove.6B.50d.txt

now=$(date "+%s")
time8=$((now-start))

echo "#9 start example test for inceptionv1 training"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
   --master local[4] \
   --driver-memory 10g \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/inception/inception.py \
   --maxIteration 20 \
   -b 8 \
   -f ${HDFS_URI}/imagenet-mini
now=$(date "+%s")
time9=$((now-start))

echo "#10 start example test for pytorch"
#timer
start=$(date "+%s")
echo "start example test for pytorch SimpleTrainingExample"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
   --master local[1] \
   --driver-memory 5g \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/SimpleTrainingExample.py

echo "start example test for pytorch mnist training"
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
   --master local[1] \
   --driver-memory 5g \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/Lenet_mnist.py
now=$(date "+%s")
time10=$((now-start))

echo "#11 start example test for openvino"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-models/faster_rcnn_resnet101_coco.xml ]
then
   echo "analytics-zoo-models/faster_rcnn_resnet101_coco already exists."
else
   wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.xml \
    -P analytics-zoo-models
   wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.bin \
    -P analytics-zoo-models
fi
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data
fi
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 10g \
    --executor-memory 10g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/openvino/predict.py \
    --image analytics-zoo-data/data/object-detection-coco \
    --model analytics-zoo-models/faster_rcnn_resnet101_coco.xml
now=$(date "+%s")
time11=$((now-start))

echo "#12 start example for vnni/openvino"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-models/vnni ]
then
   echo "analytics-zoo-models/resnet_v1_50.xml already exists."
else
   wget -nv $FTP_URI/analytics-zoo-models/openvino/vnni/resnet_v1_50.zip \
    -P analytics-zoo-models
    unzip -q analytics-zoo-models/resnet_v1_50.zip -d analytics-zoo-models/vnni
fi
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data
fi
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/vnni/openvino/predict.py \
    --model analytics-zoo-models/vnni/resnet_v1_50.xml \
    --image analytics-zoo-data/data/object-detection-coco
now=$(date "+%s")
time12=$((now-start))

echo "#13 start example test for streaming Object Detection"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
fi

if [ -f analytics-zoo-models/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model ]
then
    echo "analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model already exists"
else
    wget -nv $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
     -P analytics-zoo-models
fi

mkdir -p output
mkdir -p stream
while true
do
   temp1=$(find analytics-zoo-data/data/object-detection-coco -type f|wc -l)
   temp2=$(find ./output -type f|wc -l)
   temp3=$(($temp1+$temp1))
   if [ $temp3 -eq $temp2 ];then
       kill -9 $(ps -ef | grep streaming_object_detection | grep -v grep |awk '{print $2}')
       rm -r output
       rm -r stream
   break
   fi
done  &
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/objectdetection/streaming_object_detection.py \
    --streaming_path ./stream \
    --model analytics-zoo-models/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
    --output_path ./output  &
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/objectdetection/image_path_writer.py \
    --streaming_path ./stream \
    --img_path analytics-zoo-data/data/object-detection-coco

now=$(date "+%s")
time13=$((now-start))

echo "#14 start example test for streaming Text Classification"
if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/streaming/text-model.zip -P analytics-zoo-data/data/streaming/
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/
fi
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 5g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/textclassification/streaming_text_classification.py \
    --model analytics-zoo-data/data/streaming/text-model/text_classifier.model \
    --index_path analytics-zoo-data/data/streaming/text-model/word_index.txt \
    --input_file analytics-zoo-data/data/streaming/text-model/textfile/ > 1.log &
while :
do
echo "I am strong and I am smart" >> analytics-zoo-data/data/streaming/text-model/textfile/s
if [ -n "$(grep "top-5" 1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep streaming_text_classification | grep -v grep |awk '{print $2}')
    rm 1.log
    sleep 1s
    break
fi
done
now=$(date "+%s")
time14=$((now-start))

echo "#15 start example test for attention"
#timer
start=$(date "+%s")
sed "s/max_features = 20000/max_features = 100/g;s/max_len = 200/max_len = 10/g;s/hidden_size=128/hidden_size=8/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/transformer.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --conf spark.executor.extraJavaOptions="-Xss512m" \
    --conf spark.driver.extraJavaOptions="-Xss512m" \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 100g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py

now=$(date "+%s")
time15=$((now-start))
echo "#15 attention time used:$time15 seconds"

echo "#1 textclassification time used: $time1 seconds"
echo "#2 autograd time used: $time2 seconds"
echo "#3 image-classification time used: $time3 seconds"
echo "#4 object-detection loss and layer time used: $time4 seconds"
echo "#5 nnframes time used: $time5 seconds"
echo "#6 tensorflow time used: $time6 seconds"
echo "#7 anomalydetection time used: $time7 seconds"
echo "#8 qaranker time used: $time8 seconds"
echo "#9 inceptionV1 training time used: $time9 seconds"
echo "#10 pytorch time used: $time10 seconds"
echo "#11 openvino time used: $time11 seconds"
echo "#12 vnni/openvino time used: $time12 seconds"
echo "#13 streaming Object Detection time used: $time13 seconds"
echo "#14 streaming text classification time used: $time14 seconds"
echo "#15 attention time used:$time15 seconds"

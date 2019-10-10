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
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data 
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi 
if [ -f analytics-zoo-data/data/20news-18828.tar.gz ]
then 
    echo "analytics-zoo-data/data/20news-18828.tar.gz already exists" 
else
    wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P analytics-zoo-data/data 
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

echo "#2 start example test for customized loss and layer (Funtional API)"
#timer
start=$(date "+%s")
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/custom.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/custom.py \
    --nb_epoch 2
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

if [ -f analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model ]
then
    echo "analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model\
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
    -f hdfs://172.168.2.181:9000/kaggle/train_100 \
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
    wget $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model \
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
    analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model hdfs://172.168.2.181:9000/kaggle/train_100 /tmp
now=$(date "+%s")
time4=$((now-start))

echo "#5 start example test for nnframes"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]
then
   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
   wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi
 if [ -f analytics-zoo-data/data/dogs-vs-cats/train.zip ]
then
   echo "analytics-zoo-data/data/dogs-vs-cats/train.zip already exists."
else
   # echo "Downloading dogs and cats images"
   wget  $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip\
    -P analytics-zoo-data/data/dogs-vs-cats
   unzip analytics-zoo-data/data/dogs-vs-cats/train.zip -d analytics-zoo-data/data/dogs-vs-cats
   mkdir -p analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/cat.71* analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/dog.71* analytics-zoo-data/data/dogs-vs-cats/samples
   # echo "Finished downloading images"
fi
# total batch size: 32 should be divided by total core number: 28
sed "s/setBatchSize(32)/setBatchSize(56)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/image_finetuning_example.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/tmp.py
sed "s/setBatchSize(32)/setBatchSize(56)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/tmp.py
sed "s/setBatchSize(4)/setBatchSize(56)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/ImageInferenceExample.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/tmp.py
    
echo "start example test for nnframes finetune"
${SPARK_HOME}/bin/spark-submit \
   --master local[2] \
   --driver-memory 10g \
   --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/tmp.py \
   --jars ${ANALYTICS_ZOO_JAR} \
   --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
   --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/image_finetuning_example.py \
   analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model analytics-zoo-data/data/dogs-vs-cats/samples
   
echo "start example test for nnframes imageInference"
${SPARK_HOME}/bin/spark-submit \
   --master local[1] \
   --driver-memory 3g \
   --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/tmp.py \
   --jars ${ANALYTICS_ZOO_JAR} \
   --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
   --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/tmp.py \
   analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model hdfs://172.168.2.181:9000/kaggle/train_100

echo "start example test for nnframes imageTransferLearning"
${SPARK_HOME}/bin/spark-submit \
   --master local[1] \
   --driver-memory 5g \
   --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/tmp.py \
   --jars ${ANALYTICS_ZOO_JAR} \
   --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
   --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
   ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/tmp.py\
   analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model analytics-zoo-data/data/dogs-vs-cats/samples
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
   wget $FTP_URI/analytics-zoo-models/tensorflow/ssd_mobilenet_v1_coco_2017_11_17.tar.gz \
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
    --image hdfs://172.168.2.181:9000/kaggle/train_100 \
    --model analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb

echo "start example test for tensorflow distributed_training"
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
   
   wget $FTP_URI/analytics-zoo-tensorflow-models/models/research/slim.tar.gz -P analytics-zoo-tensorflow-models
   tar -zxvf analytics-zoo-tensorflow-models/slim.tar.gz -C analytics-zoo-tensorflow-models
   
   echo "Finished downloading research/slim"
   export PYTHONPATH=`pwd`/analytics-zoo-tensorflow-models/slim:$PYTHONPATH
 fi

echo "start example test for TFPark tf_optimizer train_lenet 1"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train_lenet.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train_lenet.py 1 1000\

sed "s%/tmp%analytics-zoo-tensorflow-models%g;s%models/slim%slim%g"
if [ -d analytics-zoo-tensorflow-models/slim ]
then
    echo "analytics-zoo-tensorflow-models/slim already exists."
else
    echo "Downloading research/slim"
   
   wget $FTP_URI/analytics-zoo-tensorflow-models/models/research/slim.tar.gz -P analytics-zoo-tensorflow-models
   tar -zxvf analytics-zoo-tensorflow-models/slim.tar.gz -C analytics-zoo-tensorflow-models
   
   echo "Finished downloading research/slim"
   export PYTHONPATH=`pwd`/analytics-zoo-tensorflow-models/slim:$PYTHONPATH
fi

echo "start example test for tensorflow distributed_training evaluate_lenet 2"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate_lenet.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate_lenet.py 1000\

echo "start example test for tensorflow distributed_training train_mnist_keras 3"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train_mnist_keras.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train_mnist_keras.py 1 1000\

echo "start example test for tensorflow distributed_training evaluate_lenet 4"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate_mnist_keras.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate_mnist_keras.py 1000\

    
now=$(date "+%s")
time6=$((now-start))
echo "#6 tensorflow time used:$time6 seconds"

echo "#7 start example test for anomalydetection"
if [ -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv ]
then
    echo "analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
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
echo "#7 anomalydetection time used:$time7 seconds"

echo "#8 start example test for qaranker"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi
if [ -f analytics-zoo-data/data/WikiQAProcessed.zip ]
then
    echo "analytics-zoo-data/data/WikiQAProcessed.zip already exists"
else
    wget https://s3.amazonaws.com/analytics-zoo-data/WikiQAProcessed.zip -P analytics-zoo-data/data
    unzip analytics-zoo-data/data/WikiQAProcessed.zip -d analytics-zoo-data/data/
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
echo "#1 textclassification time used:$time1 seconds"
echo "#2 customized loss and layer time used:$time2 seconds"
echo "#3 image-classification time used:$time3 seconds"
echo "#4 object-detection loss and layer time used:$time4 seconds"
echo "#5 nnframes time used:$time5 seconds"
echo "#6 tensorflow time used:$time6 seconds"
echo "#7 anomalydetection time used:$time7 seconds"
echo "#8 qaranker time used:$time8 seconds"

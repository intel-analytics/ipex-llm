#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_HOME
export ANALYTICS_ZOO_HOME_DIST=$ANALYTICS_ZOO_HOME/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME_DIST}/conf/spark-bigdl.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

echo "#1 start example test for textclassification"
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

echo "check if model directory exists"
if [ ! -d analytics-zoo-models ]
then
    mkdir analytics-zoo-models
fi
if [ -f analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0 ]
then
    echo "analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0 already exists"
else
    wget $FTP_URI/analytics-zoo-models/imageclassification/imagenet/analytics-zoo_squeezenet_imagenet_0.1.0 \
    -P analytics-zoo-models
fi
if [ -f analytics-zoo-models-new/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model ]
then
    echo "analytics-zoo-models-new/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models-new/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model \
    -P analytics-zoo-models-new
fi

${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --nb_epoch 2 \
    --data_path analytics-zoo-data/data

echo "#2 start example test for customized loss and layer (Funtional API)"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/autograd/custom.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/autograd/custom.py \
    --nb_epoch 2

${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/imageclassification/predict.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/imageclassification/predict.py \
    -f hdfs://172.168.2.181:9000/kaggle/train_100 \
    --model analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0 \
    --topN 5

${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/objectdetection/predict.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/objectdetection/predict.py \
    analytics-zoo-models-new/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model hdfs://172.168.2.181:9000/kaggle/train_100 /tmp

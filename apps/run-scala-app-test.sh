#!/bin/bash

echo "App[Model-inference-example] Test"
echo "# Test 1 text-classification-training"

cd ${BIGDL_ROOT}/apps/model-inference-examples/
mkdir "models"

if [ -d analytics-zoo-data/data/ ]
then
    echo "analytics-zoo-data/data/ already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data/glove
    unzip -q analytics-zoo-data/data/glove/glove.6B.zip -d analytics-zoo-data/data/glove/glove
    wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P analytics-zoo-data/data/news20/
    tar -zxvf analytics-zoo-data/data/news20/20news-18828.tar.gz -C analytics-zoo-data/data/news20/
fi

cd text-classification-training
mvn clean
mvn clean package
mvn install
#return model-inference-examples/
cd ${BIGDL_ROOT}/apps/model-inference-examples/

#timer
start=$(date "+%s")

${SPARK_HOME}/bin/spark-shell \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --jars ./text-classification-training/target/text-classification-training-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --conf spark.executor.memory="20g" \
    --class com.intel.analytics.bigdl.apps.textclassfication.training.TextClassificationTrainer \
    --batchSize 2000 --nbEpoch 2 \
    --trainDataDir "analytics-zoo-data/data/news20/20news-18828" \
    --embeddingFile "analytics-zoo-data/data/glove/glove/glove.6B.300d.txt" \
    --modelSaveDirPath "models/text-classification.bigdl"

now=$(date "+%s")
time1=$((now-start))
echo "#App[Model-inference-example] Test 1: text-classification-training time used:$time1 seconds"

echo "# Test Apps -- 2.text-classification-inference"

cd ${BIGDL_ROOT}/apps/model-inference-examples/text-classification-inference
mvn clean
mvn clean package

echo "# Test 2.1 text-classification-inference:SimpleDriver"
#timer
start=$(date "+%s")

java -cp target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DEMBEDDING_FILE_PATH=${BIGDL_ROOT}/apps/model-inference-examples/analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=${BIGDL_ROOT}/apps/model-inference-examples/models/text-classification.bigdl \
com.intel.analytics.bigdl.apps.textclassfication.inference.SimpleDriver

now=$(date "+%s")
time2=$((now-start))
echo "#App[Model-inference-example] Test 3.1: text-classification-inference:SimpleDriver time used:$time2 seconds"

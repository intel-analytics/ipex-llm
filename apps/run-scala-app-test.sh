#!/bin/bash

echo "App[Model-inference-example] Test"
echo "# Test 1 text-classification-training"

cd ${BIGDL_ROOT}/apps/model-inference-examples/
mkdir "models"

if [ -d tmp/data/ ]
then
    echo "analytics-zoo-data/data/ already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P tmp/data
    unzip -q tmp/data/object-detection-coco.zip -d tmp/data/
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P tmp/data/glove
    unzip -q tmp/data/glove/glove.6B.zip -d tmp/data/glove/glove
    wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P tmp/data/news20/
    tar -zxvf tmp/data/news20/20news-18828.tar.gz -C tmp/data/news20/
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
    --trainDataDir "tmp/data/news20/20news-18828" \
    --embeddingFile "tmp/data/glove/glove/glove.6B.300d.txt" \
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
-DEMBEDDING_FILE_PATH=${BIGDL_ROOT}/apps/model-inference-examples/tmp/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=${BIGDL_ROOT}/apps/model-inference-examples/models/text-classification.bigdl \
com.intel.analytics.bigdl.apps.textclassfication.inference.SimpleDriver

now=$(date "+%s")
time2=$((now-start))
echo "#App[Model-inference-example] Test 3.1: text-classification-inference:SimpleDriver time used:$time2 seconds"

# echo "# Test 2.2 text-classification-inference:WebServiceDriver"
# #timer
# start=$(date "+%s")

# mvn spring-boot:run -DEMBEDDING_FILE_PATH=${BIGDL_ROOT}/apps/model-inference-examples/tmp/data/glove/glove/glove.6B.300d.txt \
# -DMODEL_PATH=${BIGDL_ROOT}/apps/model-inference-examples/models/text-classification.bigdl &
# while :
# do
#   curl -d hello -x "" http://localhost:8080/predict > 1.log &
# if [ -n "$(grep "class" ${BIGDL_ROOT}/apps/model-inference-examples/text-classification-inference/1.log)" ];then
#     echo "----Find-----"
#     kill -9 $(ps -ef | grep text-classification | grep -v grep |awk '{print $2}')
#     rm 1.log
#     sleep 1s
#     break
# fi
# done

# now=$(date "+%s")
# time3=$((now-start))
# echo "#App[Model-inference-example] Test 3.2: text-classification-inference:WebServiceDriver time used:$time3 seconds"

echo "# Test 3.recommendation-inference"

#recommendation
cd ${BIGDL_ROOT}/apps/model-inference-examples/recommendation-inference
mvn clean
mvn clean package
cd ${BIGDL_ROOT}/apps/model-inference-examples

if [ -f models/recommendation/ncf.bigdl ]
then
    echo "models/recommedation/ncf.bigdl already exists"
else
    wget $FTP_URI/analytics-zoo-models/recommendation/ncf.bigdl -P models/recommendation/
fi
echo "# Test 3.1 recommendation-inference:SimpleScalaDriver"
#timer
start=$(date "+%s")

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DMODEL_PATH=./models/recommendation/ncf.bigdl \
com.intel.analytics.bigdl.apps.recommendation.inference.SimpleScalaDriver

now=$(date "+%s")
time4=$((now-start))
echo "#App[Model-inference-example] Test 3.1: recommendation-inference:SimpleScalaDriver time used:$time4 seconds"

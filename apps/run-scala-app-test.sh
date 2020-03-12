#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_JAR_AND_SPARK=`find ${ANALYTICS_ZOO_ROOT}/zoo/target -type f -name "analytics-zoo*jar-with-dependencies-and-spark.jar"`

echo "#1 start fraud-detection scala app test"
#timer
start=$(date "+%s")

FILENAME="${ANALYTICS_ZOO_HOME}/apps/fraud-detection/creditcard.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
	echo "Downloading creditcard.csv"
	wget -P ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/ $FTP_URI/analytics-zoo-data/apps/fraudDetection/creditcard.csv
	echo "Finished"
fi

#convert notebook to scala script
jupyter nbconvert --to script ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/fraud-detection.ipynb

#add object frauddetection extends App{}
sed -i '/import com.intel.analytics.bigdl.utils.LoggerFilter/ a\ object frauddetection extends App{' ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/fraud-detection.scala
sed -i '/evaluateModel(predictDF)/ a\ }' ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/fraud-detection.scala
sed -i 's@"/tmp/datasets/creditcard.csv"@sys.env("ANALYTICS_ZOO_HOME")+"/apps/fraud-detection/creditcard.csv"@g' ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/fraud-detection.scala

mkdir ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/bin

#compile scala to .class
scalac -classpath "$ANALYTICS_ZOO_JAR_AND_SPARK:${ANALYTICS_ZOO_HOME}/apps/fraud-detection/fraud-1.0.1-SNAPSHOT.jar:$SPARK_HOME/jars/*" \
	-d ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/bin \
    ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/fraud-detection.scala
echo "compile scala to .class finished"

#pack classes to fraud.jar
cd ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/bin
jar cvf fraud.jar *
echo "pack classes to fraud.jar finished"

master=local[*]
$SPARK_HOME/bin/spark-submit \
	--verbose --master $master \
    --conf spark.executor.cores=1 \
    --total-executor-cores 4 \
    --driver-memory 200g \
    --executor-memory 200g \
    --class frauddetection \
    --jars $ANALYTICS_ZOO_JAR_AND_SPARK,${ANALYTICS_ZOO_HOME}/apps/fraud-detection/fraud-1.0.1-SNAPSHOT.jar \
    ${ANALYTICS_ZOO_HOME}/apps/fraud-detection/bin/fraud.jar

now=$(date "+%s")
time1=$((now-start))
echo "#1 fraud-detection time used:$time1 seconds"

echo "App[Model-inference-example] Test"
echo "# Test 2 text-classification-training"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/
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
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--jars ./text-classification-training/target/text-classification-training-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--conf spark.executor.memory="20g" \
--class com.intel.analytics.zoo.apps.textclassfication.training.TextClassificationTrainer \
--batchSize 2000 --nbEpoch 2 \
--trainDataDir "analytics-zoo-data/data/news20/20news-18828" \
--embeddingFile "analytics-zoo-data/data/glove/glove/glove.6B.300d.txt" \
--modelSaveDirPath "models/text-classification.bigdl"

now=$(date "+%s")
time2=$((now-start))
echo "#App[Model-inference-example] Test 2: text-classification-training time used:$time2 seconds"

echo "# Test Apps -- 3.text-classification-inference"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-inference
mvn clean
mvn clean package

echo "# Test 3.1 text-classification-inference:SimpleDriver"
#timer
start=$(date "+%s")

java -cp target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DEMBEDDING_FILE_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/models/text-classification.bigdl \
com.intel.analytics.zoo.apps.textclassfication.inference.SimpleDriver

now=$(date "+%s")
time3=$((now-start))
echo "#App[Model-inference-example] Test 3.1: text-classification-inference:SimpleDriver time used:$time3 seconds"

echo "# Test 3.2 text-classification-inference:WebServiceDriver"
#timer
start=$(date "+%s")

mvn spring-boot:run -DEMBEDDING_FILE_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/models/text-classification.bigdl &
while :
do
  curl -d hello -x "" http://localhost:8080/predict > 1.log &
if [ -n "$(grep "class" ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-inference/1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep text-classification | grep -v grep |awk '{print $2}')
    rm 1.log
    sleep 1s
    break
fi
done

now=$(date "+%s")
time4=$((now-start))
echo "#App[Model-inference-example] Test 3.2: text-classification-inference:WebServiceDriver time used:$time4 seconds"

echo "# Test 4.recommendation-inference"

#recommendation
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/recommendation-inference
mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

if [ -f analytics-zoo-models/recommendation/ncf.bigdl ]
then
    echo "analytics-zoo-models/recommedation/ncf.bigdl already exists"
else
    wget $FTP_URI/analytics-zoo-models/recommendation/ncf.bigdl -P analytics-zoo-models/recommendation/
fi
echo "# Test 4.1 recommendation-inference:SimpleScalaDriver"
#timer
start=$(date "+%s")

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DMODEL_PATH=./analytics-zoo-models/recommendation/ncf.bigdl \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleScalaDriver

now=$(date "+%s")
time5=$((now-start))
echo "#App[Model-inference-example] Test 4.1: recommendation-inference:SimpleScalaDriver time used:$time5 seconds"

echo "# Test 4.2 recommendation-inference:SimpleDriver[Java]"
#timer
start=$(date "+%s")

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DMODEL_PATH=./analytics-zoo-models/recommendation/ncf.bigdl \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleDriver

now=$(date "+%s")
time6=$((now-start))
echo "#App[Model-inference-example] Test 4.2: recommendation-inference:SimpleDriver time used:$time6 seconds"

echo "# Test 5.model-inference-flink"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/model-inference-flink
mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

if [ -f ./flink-1.7.2/bin/start-cluster.sh ]
then
    echo "flink-1.7.2/bin/start-cluster.sh already exists"
else
    wget $FTP_URI/flink-1.7.2.zip
    unzip flink-1.7.2.zip
fi

if [ -f analytics-zoo-data/data/streaming/text-model/2.log ]
then
    echo "analytics-zoo-data/data/streaming/text-model/2.log already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/streaming/text-model/2.log -P analytics-zoo-data/data/streaming/text-model/2.log
fi

./flink-1.7.2/bin/start-cluster.sh

echo "# Test 5.1 model-inference-flink:Text Classification"
#timer
start=$(date "+%s")

./flink-1.7.2/bin/flink run \
./model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--inputFile analytics-zoo-data/data/streaming/text-model/2.log \
--embeddingFilePath analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
--modelPath models/text-classification.bigdl \
--parallelism 1

now=$(date "+%s")
time7=$((now-start))
echo "#App[Model-inference-example] Test 5.1: model-inference-flink:Text Classification time used:$time7 seconds"

./flink-1.7.2/bin/stop-cluster.sh

if [ -f mobilenet_v1_1.0_224_frozen.pb ]
then
    echo "analytics-zoo-models/flink_model/mobilenet_v1_1.0_224_frozen.pb already exists"
else
    wget ${FTP_URI}/analytics-zoo-models/flink_model/mobilenet_v1_1.0_224_frozen.pb
fi

./flink-1.7.2/bin/start-cluster.sh

echo "# Test 5.2 model-inference-flink: Image Classification"
#timer
start=$(date "+%s")

./flink-1.7.2/bin/flink run \
-m localhost:8081 -p 1 \
-c com.intel.analytics.zoo.apps.model.inference.flink.ImageClassification.ImageClassificationStreaming  \
${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
--modelPath mobilenet_v1_1.0_224_frozen.pb   --modelType frozenModel   \
--images ${ANALYTICS_ZOO_ROOT}/zoo/src/test/resources/imagenet/n04370456/ \
--classes ${ANALYTICS_ZOO_ROOT}/zoo/src/main/resources/imagenet_classname.txt

now=$(date "+%s")
time8=$((now-start))
echo "#App[Model-inference-example] Test 5.1: model-inference-flink: Image Classification time used:$time8 seconds"

./flink-1.7.2/bin/stop-cluster.sh

echo "#1 fraud-detection time used:$time1 seconds"
echo "#2 text-classification-training time used:$time2 seconds"
echo "#3.1 text-classification-inference:SimpleDriver time used:$time3 seconds"
echo "#3.2 text-classification-inference:WebServiceDriver time used:$time4 seconds"
echo "#4.1 recommendation-inference:SimpleScalaDriver time used:$time5 seconds"
echo "#4.2 recommendation-inference:SimpleDriver time used:$time6 seconds"
echo "#5.1 model-inference-flink:Text Classification time used:$time7 seconds"
echo "#5.2 model-inference-flink:Image Classification time used:$time8 seconds"

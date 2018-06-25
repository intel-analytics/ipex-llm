#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_HOME
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_JAR_AND_SPARK=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies-and-spark.jar"`

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
time=$((now-start))
echo "#1 fraud-detection time used:$time seconds"

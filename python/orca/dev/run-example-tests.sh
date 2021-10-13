#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=$(find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar")
export ANALYTICS_ZOO_PYZIP=$(find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip")
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH
export BIGDL_CLASSPATH=${ANALYTICS_ZOO_JAR}

set -e

echo "#15 start example test for orca data"
if [ -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv ]; then
  echo "analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
    -P analytics-zoo-data/data/NAB/nyc_taxi/
fi
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --master ${MASTER} \
  --driver-memory 2g \
  --executor-memory 2g \
  ${BIGDL_ROOT}/python/orca/example/data/spark_pandas.py \
  -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv

now=$(date "+%s")
time15=$((now - start))

echo "#16 start test for orca tf imagesegmentation"
#timer
start=$(date "+%s")
# prepare data
if [ -f analytics-zoo-data/data/carvana ]; then
  echo "analytics-zoo-data/data/carvana already exists"
else
  wget $FTP_URI/analytics-zoo-data/data/carvana/train.zip \
    -P analytics-zoo-data/data/carvana/
  wget $FTP_URI/analytics-zoo-data/data/carvana/train_masks.zip \
    -P analytics-zoo-data/data/carvana/
  wget $FTP_URI/analytics-zoo-data/data/carvana/train_masks.csv.zip \
    -P analytics-zoo-data/data/carvana/
fi

# Run the example
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --master ${MASTER} \
  --driver-memory 3g \
  --executor-memory 3g \
  ${BIGDL_ROOT}/python/orca/example/learn/tf/image_segmentation/image_segmentation.py \
  --file_path analytics-zoo-data/data/carvana --epochs 1 --non_interactive
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca tf imagesegmentation failed"
  exit $exit_status
fi
now=$(date "+%s")
time16=$((now - start))

echo "#17 start test for orca tf transfer_learning"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --master ${MASTER} \
  --driver-memory 3g \
  --executor-memory 3g \
  ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca tf transfer_learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time17=$((now - start))

echo "#18 start test for orca tf basic_text_classification"
#timer
start=$(date "+%s")
sed "s/epochs=100/epochs=10/g" \
  ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/basic_text_classification.py \
  >${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/tmp.py
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --master ${MASTER} \
  --driver-memory 3g \
  --executor-memory 3g \
  ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/tmp.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca tf basic_text_classification failed"
  exit $exit_status
fi
now=$(date "+%s")
time18=$((now - start))

echo "#19 start test for orca bigdl attention"
#timer
start=$(date "+%s")
sed "s/max_features = 20000/max_features = 200/g;s/max_len = 200/max_len = 20/g;s/hidden_size=128/hidden_size=8/g;s/memory=\"100g\"/memory=\"20g\"/g;s/driver_memory=\"20g\"/driver_memory=\"3g\"/g" \
  ${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/transformer.py \
  >${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/tmp.py
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --conf spark.executor.extraJavaOptions="-Xss512m" \
  --conf spark.driver.extraJavaOptions="-Xss512m" \
  --master ${MASTER} \
  --driver-memory 3g \
  --executor-memory 20g \
  ${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/tmp.py
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca tf bigdl attention failed"
  exit $exit_status
fi
now=$(date "+%s")
time19=$((now - start))

echo "#20 start test for orca bigdl imageInference"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
  echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
  wget -nv $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --master ${MASTER} \
  --driver-memory 3g \
  --executor-memory 3g \
  ${BIGDL_ROOT}/python/orca/example/learn/bigdl/imageInference/imageInference.py \
  -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/kaggle/train_100
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca bigdl imageInference failed"
  exit $exit_status
fi
now=$(date "+%s")
time20=$((now - start))

# echo "#21 start test for orca inception_v1"
# start=$(date "+%s")

# ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
#   --master ${MASTER} \
#   --driver-memory 2g \
#   --executor-memory 10g \
#   ${BIGDL_ROOT}/python/orca/example/learn/tf/inception/inception.py \
#   -b 8 -f ${ANALYTICS_ZOO_ROOT}/pyzoo/test/zoo/resources/imagenet_to_tfrecord --imagenet ./imagenet 

# now=$(date "+%s")
# time21=$((now - start))


echo "#15 orca data time used:$time15 seconds"
echo "#16 orca tf imagesegmentation time used:$time16 seconds"
echo "#17 orca tf transfer_learning time used:$time17 seconds"
echo "#18 orca tf basic_text_classification time used:$time18 seconds"
echo "#19 orca bigdl attention time used:$time19 seconds"
echo "#20 orca bigdl imageInference time used:$time20 seconds"
#echo "#21 orca inception_v1 time used:$time21 seconds"


#!/bin/bash

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e
#set -x

echo "SERVING_JAR_PATH=$SERVING_JAR_PATH"
SPARK_JAR_PATH=$SPARK_HOME/jars/*
echo "SPARK_JAR_PATH=$SPARK_JAR_PATH"
JERSEY_JAR_PATH=/opt/work/jersey/*

if (( $# < 1)); then
  echo "Usage: service_type other_parameters"
  echo "Usage example: ranking -c /opt/work/config_ranking.yaml -p 8083"
  echo "Usage example: client -target localhost:8980 -dataDir wnd_user.parquet -k 50 -clientNum 4 -testNum 2"
  echo "service_type can be one of ranking, recall, recommender, recommender-http, feature, client, recall-init and feature-init."
  exit -1
fi

service=$1
params=${*:2}

echo "service=$service"
echo "params=$params"

if [ "$service" == "ranking" ]; then
    echo "Starting ranking service......"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    java -cp $SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.ranking.RankingServer $params
elif [ "$service" == "feature" ]; then
    echo "Starting feature service......"
    java -cp $SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.feature.FeatureServer $params
elif [ "$service" == "recall" ]; then
    echo "Starting recall service......"
    java -cp $SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.recall.RecallServer $params
elif [ "$service" == "recommender" ]; then
    echo "Starting recommender service......"
    java -cp $SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.recommender.RecommenderServer $params
elif [ "$service" == "recommender-http" ]; then
    echo "Starting recommender http server......"
    java -cp $JERSEY_JAR_PATH:$SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.recommender.HTTP.RecommenderHTTP $params
elif [ "$service" == "recall-init" ]; then
    echo "Starting initializing recall index......"
    java -Dspark.master=local[*] -cp $SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.nearline.recall.RecallInitializer $params
elif [ "$service" == "feature-init" ]; then
    echo "Starting loading initial features......"
    java -Dspark.master=local[*] -cp $SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.nearline.feature.FeatureInitializer $params
elif [ "$service" == "client" ]; then
    java -Dspark.master=local[*] -cp $SPARK_JAR_PATH:$SERVING_JAR_PATH com.intel.analytics.bigdl.friesian.serving.recommender.RecommenderMultiThreadClient $params
else
    echo "Unsupported service_type, service_type can be one of ranking, recall, recommender, recommender-http, feature, feature-init, recall-init and client."
fi

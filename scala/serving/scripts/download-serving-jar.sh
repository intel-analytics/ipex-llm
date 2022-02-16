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
if [ -z "${BIGDL_VERSION}" ]; then
  export BIGDL_VERSION=0.14.0-SNAPSHOT
  export SPARK_VERSION=2.4.6
  echo "You did not specify BIGDL_VERSION, will download the latest version: "$BIGDL_VERSION
fi

echo "BIGDL_VERSION is "$BIGDL_VERSION
echo "SPARK_VERSION is "$SPARK_VERSION
SPARK_MAJOR_VERSION=${SPARK_VERSION%%.[0-9]}
echo $SPARK_MAJOR_VERSION

if [[ $BIGDL_VERSION == *"SNAPSHOT"* ]]; then
  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-serving-spark_$SPARK_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-serving-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-serving-spark_$SPARK_VERSION-$NIGHTLY_VERSION-jar-with-dependencies.jar
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-serving-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-serving-spark_$SPARK_VERSION-$NIGHTLY_VERSION-http.jar

else
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-serving-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-serving-spark_$SPARK_VERSION-$BIGDL_VERSION-jar-with-dependencies.jar
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-serving-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-serving-spark_$SPARK_VERSION-$BIGDL_VERSION-http.jar
fi
echo "If download too slow or failed, please go to BigDL Serving Repo to download, link: "
echo "nightly: https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-serving-spark_$SPARK_VERSION/$BIGDL_VERSION/"
echo "release: https://repo1.maven.org/maven2/com/intel/analytics/bigdl/"

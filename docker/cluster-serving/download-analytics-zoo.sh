#!/bin/bash

#
# Copyright 2016 The Analytics-Zoo Authors.
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

echo $ANALYTICS_ZOO_VERSION
echo $BIGDL_VERSION
echo $SPARK_VERSION
SPARK_MAJOR_VERSION=${SPARK_VERSION%%.[0-9]}
echo $SPARK_MAJOR_VERSION

if [[ $ANALYTICS_ZOO_VERSION == *"SNAPSHOT"* ]]; then
  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/groups/public/com/intel/analytics/zoo/analytics-zoo-bigdl_$BIGDL_VERSION-spark_$SPARK_VERSION/$ANALYTICS_ZOO_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  wget https://oss.sonatype.org/content/groups/public/com/intel/analytics/zoo/analytics-zoo-bigdl_$BIGDL_VERSION-spark_$SPARK_VERSION/$ANALYTICS_ZOO_VERSION/analytics-zoo-bigdl_$BIGDL_VERSION-spark_$SPARK_VERSION-$NIGHTLY_VERSION-dist-all.zip
  unzip analytics-zoo-bigdl_$BIGDL_VERSION-spark_$SPARK_VERSION-$NIGHTLY_VERSION-dist-all.zip -d $ANALYTICS_ZOO_HOME
else
  wget https://repo1.maven.org/maven2/com/intel/analytics/zoo/analytics-zoo-bigdl_$BIGDL_VERSION-spark_$SPARK_VERSION/$ANALYTICS_ZOO_VERSION/analytics-zoo-bigdl_$BIGDL_VERSION-spark_$SPARK_VERSION-$ANALYTICS_ZOO_VERSION-dist-all.zip
  unzip analytics-zoo-bigdl_$BIGDL_VERSION-spark_$SPARK_VERSION-$ANALYTICS_ZOO_VERSION-dist-all.zip -d $ANALYTICS_ZOO_HOME
fi

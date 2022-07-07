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

set -x

echo $BIGDL_VERSION
echo $SPARK_VERSION

if [[ $BIGDL_VERSION == *"SNAPSHOT"* ]]; then
  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-friesian-spark_$SPARK_VERSION/$BIGDL_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  JAR_NAME=bigdl-friesian-spark_$SPARK_VERSION-$NIGHTLY_VERSION-serving.jar
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-friesian-spark_$SPARK_VERSION/$BIGDL_VERSION/$JAR_NAME -O bigdl-friesian-serving.jar
else
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-friesian-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-friesian-spark_$SPARK_VERSION-$BIGDL_VERSION-serving.jar -O bigdl-friesian-serving.jar
fi

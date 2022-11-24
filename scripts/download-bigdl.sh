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

if [ -z "$BIGDL_VERSION" ]; then
  echo "Please specify BIGDL_VERSION, e.g. 2.1.0, 2.2.0-SNAPSHOT."
  exit 1
fi

echo $BIGDL_VERSION

if [ -z "$SPARK_VERSION" ]; then
  echo "Please specify SPARK_VERSION, e.g. 2.4.6, 3.1.2 and 3.1.3."
  exit 1
fi

echo $SPARK_VERSION

if [[ $BIGDL_VERSION == *"SNAPSHOT"* ]]; then
  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-assembly-spark_$SPARK_VERSION/$BIGDL_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-assembly-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-assembly-spark_$SPARK_VERSION-$NIGHTLY_VERSION-fat-jars.zip
  unzip bigdl-assembly-spark_$SPARK_VERSION-$NIGHTLY_VERSION-fat-jars.zip -d $BIGDL_HOME
else
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-assembly-spark_$SPARK_VERSION-$BIGDL_VERSION-fat-jars.zip
  unzip bigdl-assembly-spark_$SPARK_VERSION-$BIGDL_VERSION-fat-jars.zip -d $BIGDL_HOME
fi

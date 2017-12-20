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

#echo $BIGDL_VERSION
#echo $SPARK_VERSION

if [[ $BIGDL_VERSION == *"SNAPSHOT"* ]]; then
  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/dist-spark-$SPARK_VERSION-scala-2.11.8-all/$BIGDL_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/dist-spark-$SPARK_VERSION-scala-2.11.8-all/$BIGDL_VERSION/dist-spark-$SPARK_VERSION-scala-2.11.8-all-$NIGHTLY_VERSION-dist.zip
  unzip dist-spark-$SPARK_VERSION-scala-2.11.8-all-$NIGHTLY_VERSION-dist.zip -d $BIGDL_HOME
else
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/dist-spark-$SPARK_VERSION-scala-2.11.8-all/$BIGDL_VERSION/dist-spark-$SPARK_VERSION-scala-2.11.8-all-$BIGDL_VERSION-dist.zip
  unzip dist-spark-$SPARK_VERSION-scala-2.11.8-all-$BIGDL_VERSION-dist.zip -d $BIGDL_HOME  
fi

#!/usr/bin/env bash

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

#
# For internal use to deploy bigdl maven artifacts from jenkins server to sonatype
#

set -e

# Check java
if type -p java>/dev/null; then
    _java=java
else
    echo "Java is not installed"
    exit 1
fi

# We deploy artifacts build from jdk 1.7
JDK_VERSION=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
if [[ "$JDK_VERSION" > "1.8" ]]; then
    echo Require a jdk 1.7 version
    exit 1
fi
if [[ "$JDK_VERSION" < "1.7" ]]; then
    echo Require a jdk 1.7 version
    exit 1
fi
export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m -XX:MaxPermSize=1G"

# Check if mvn installed
MVN_INSTALL=$(which mvn 2>/dev/null | grep mvn | wc -l)
if [ $MVN_INSTALL -eq 0 ]; then
  echo "MVN is not installed. Exit"
  exit 1
fi

# Remove dist module
mv spark/pom.xml spark/pom.xml.origin
cat spark/pom.xml.origin | sed 's/<module>dist<\/module>//' > spark/pom.xml

# Backup the origin bigdl pom.xml
mv spark/dl/pom.xml spark/dl/pom.xml.origin

# Deploy common modules and bigdl-1.5
cat spark/dl/pom.xml.origin | sed 's/<artifactId>bigdl<\/artifactId>/<artifactId>bigdl-SPARK_1.5<\/artifactId>/' > spark/dl/pom.xml
mvn clean deploy -DskipTests -P sign -Dspark.version=1.5.2
# Upload package to download for bigdl-1.5
cd spark/dist/
mvn clean deploy -DskipTests -P sign -Dspark.version=1.5.2
cd ../../
# Upload package to download for bigdl-1.5 on mac
mvn clean package -DskipTests -P mac -Dspark.version=1.5.2
cd spark/dist/
mvn clean deploy -DskipTests -P sign -P mac -Dspark.version=1.5.2
cd ../../


# Deploy bigdl-1.6
cat spark/dl/pom.xml.origin | sed 's/<artifactId>bigdl<\/artifactId>/<artifactId>bigdl-SPARK_1.6<\/artifactId>/' > spark/dl/pom.xml
mvn clean install -DskipTests -P spark_1.6 -Dspark.version=1.6.3
cd spark/dl/
mvn clean deploy -DskipTests -P sign -P spark_1.6 -Dspark.version=1.6.3
# Upload package to download for bigdl-1.6
cd ../dist/
mvn clean deploy -DskipTests -P sign -P spark_1.6 -Dspark.version=1.6.3
cd ../../
# Upload package to download for bigdl-1.6 on mac
mvn clean package -DskipTests -P mac -P spark_1.6 -Dspark.version=1.6.3
cd spark/dist/
mvn clean deploy -DskipTests -P sign -P mac -P spark_1.6 -Dspark.version=1.6.3
cd ../../

# Deploy spark-2.0 project
cd spark/spark-version/2.0/
mvn clean deploy -DskipTests -P sign -P spark_2.0
cd ../../../

# Deploy bigdl-2.0
cat spark/dl/pom.xml.origin | sed 's/<artifactId>bigdl<\/artifactId>/<artifactId>bigdl-SPARK_2.0<\/artifactId>/' > spark/dl/pom.xml
mvn clean install -DskipTests -P spark_2.0 -Dspark.version=2.0.2
cd spark/dl/
mvn clean deploy -DskipTests -P sign -P spark_2.0 -Dspark.version=2.0.2
# Upload package to download for bigdl-2.0
cd ../dist/
mvn clean deploy -DskipTests -P sign -P spark_2.0 -Dspark.version=2.0.2
cd ../../
# Upload package to download for bigdl-2.0 on mac
mvn clean package -DskipTests -P mac -P spark_2.0 -Dspark.version=2.0.2
cd spark/dist/
mvn clean deploy -DskipTests -P sign -P mac -P spark_2.0 -Dspark.version=2.0.2
cd ../../

# Deploy bigdl-2.1
cat spark/dl/pom.xml.origin | sed 's/<artifactId>bigdl<\/artifactId>/<artifactId>bigdl-SPARK_2.1<\/artifactId>/' > spark/dl/pom.xml
mvn clean install -DskipTests -P spark_2.1 -Dspark.version=2.1.1
cd spark/dl/
mvn clean deploy -DskipTests -P sign -P spark_2.1 -Dspark.version=2.1.1
# Upload package to download for bigdl-2.1
cd ../dist/
mvn clean deploy -DskipTests -P sign -P spark_2.1 -Dspark.version=2.1.1
cd ../../
# Upload package to download for bigdl-2.1 on mac
mvn clean package -DskipTests -P mac -P spark_2.1 -Dspark.version=2.1.1
cd spark/dist/
mvn clean deploy -DskipTests -P sign -P mac -P spark_2.1 -Dspark.version=2.1.1
cd ../../


mv spark/pom.xml.origin spark/pom.xml
mv spark/dl/pom.xml.origin spark/dl/pom.xml

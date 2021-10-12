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

# We deploy artifacts build from jdk 1.8
JDK_VERSION=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
if [[ "$JDK_VERSION" > "1.9" ]]; then
    echo Require a jdk 1.8 version
    exit 1
fi
if [[ "$JDK_VERSION" < "1.8" ]]; then
    echo Require a jdk 1.8 version
    exit 1
fi
export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"

# Check if mvn installed
MVN_INSTALL=$(which mvn 2>/dev/null | grep mvn | wc -l)
if [ $MVN_INSTALL -eq 0 ]; then
  echo "MVN is not installed. Exit"
  exit 1
fi

# Append Spark platform variable to bigdl artifact name and spark-version artifact name
mv dllib/pom.xml dllib/pom.xml.origin
cat dllib/pom.xml.origin | sed 's/<artifactId>bigdl-dllib<\/artifactId>/<artifactId>bigdl-dllib-${SPARK_PLATFORM}<\/artifactId>/' | \
       	sed 's/<artifactId>${spark-version.project}<\/artifactId>/<artifactId>${spark-version.project}-${SPARK_PLATFORM}<\/artifactId>/' > dllib/pom.xml

mv common/spark-version/2.0/pom.xml common/spark-version/2.0/pom.xml.origin
cat common/spark-version/2.0/pom.xml.origin | sed 's/<artifactId>2.0<\/artifactId>/<artifactId>2.0-${SPARK_PLATFORM}<\/artifactId>/' > common/spark-version/2.0/pom.xml

mv orca/pom.xml orca/pom.xml.origin
cat orca/pom.xml.origin | sed 's/<artifactId>bigdl-orca<\/artifactId>/<artifactId>bigdl-orca-${SPARK_PLATFORM}<\/artifactId>/' > orca/pom.xml

function deploy {
    mvn clean install -DskipTests -P sign -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3 ${*:4}
    cd common/spark-version && mvn deploy -DskipTests -P sign -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3 ${*:4} && cd ../..
    cd dllib && mvn deploy -DskipTests -P sign -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3 ${*:4} && cd ..
    cd orca && mvn deploy -DskipTests -P sign -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3 ${*:4} && cd ..
}

deploy 2.1.1 SPARK_2.1 '-P spark_2.x' -pl '!friesian'
deploy 2.2.0 SPARK_2.2 '-P spark_2.x' -pl '!friesian'
deploy 2.3.1 SPARK_2.3 '-P spark_2.x' -pl '!friesian'
deploy 2.4.6 SPARK_2.4 '-P spark_2.x'

mv dllib/pom.xml.origin dllib/pom.xml
mv common/spark-version/2.0/pom.xml.origin common/spark-version/2.0/pom.xml
mv orca/pom.xml.origin orca/pom.xml

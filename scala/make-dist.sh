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
# Create a folder contain all files for dist
#

set -e

RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR

# Check java
if type -p java>/dev/null; then
    _java=java
else
    echo "Java is not installed"
    exit 1
fi

MVN_OPTS_LIST="-Xmx2g -XX:ReservedCodeCacheSize=512m"

if [[ "$_java" ]]; then
    version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    if [[ "expr $version" < "1.7" ]]; then
        echo Require a java version not lower than 1.7
        exit 1
    fi
    # For jdk7
    if [[ "expr $version" < "1.8" ]]; then
        MVN_OPTS_LIST="$MVN_OPTS_LIST -XX:MaxPermSize=1G"
    fi
fi

export MAVEN_OPTS=${MAVEN_OPTS:-"$MVN_OPTS_LIST"}

# Check if mvn installed
MVN_INSTALL=$(which mvn 2>/dev/null | grep mvn | wc -l)
if [ $MVN_INSTALL -eq 0 ]; then
  echo "MVN is not installed. Exit"
  exit 1
fi

mvn -f $RUN_SCRIPT_DIR package -DskipTests $*

BASEDIR=$(dirname "$0")
DIST_DIR=$BASEDIR/../dist/

if [ ! -d "$DIST_DIR" ]
then
  mkdir -p $DIST_DIR/lib
  mkdir -p $DIST_DIR/conf
  mkdir -p $DIST_DIR/bin
else
  rm -r $DIST_DIR
  mkdir -p $DIST_DIR/lib
  mkdir -p $DIST_DIR/conf
  mkdir -p $DIST_DIR/bin
fi

cp -r $BASEDIR/dllib/target/bigdl-dllib*-jar-with-dependencies.jar $DIST_DIR/lib
cp -r $BASEDIR/dllib/target/bigdl-dllib*-python-api.zip $DIST_DIR/lib
cp -r $BASEDIR/orca/target/bigdl-orca*-jar-with-dependencies.jar $DIST_DIR/lib
cp -r $BASEDIR/orca/target/bigdl-orca*-python-api.zip $DIST_DIR/lib
if [ -f $BASEDIR/friesian/target/bigdl-friesian*-jar-with-dependencies.jar ]; then
    cp -r $BASEDIR/friesian/target/bigdl-friesian*-jar-with-dependencies.jar $DIST_DIR/lib
fi
if [ -f $BASEDIR/friesian/target/bigdl-friesian*-python-api.zip ]; then
    cp -r $BASEDIR/friesian/target/bigdl-friesian*-python-api.zip $DIST_DIR/lib
fi
cp -r $BASEDIR/dllib/src/main/resources/spark-bigdl.conf $DIST_DIR/conf
cp -r $BASEDIR/../scripts/* $DIST_DIR/bin

#!/usr/bin/env bash

#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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

# Check java
if type -p java>/dev/null; then
    _java=java
else
    echo "Java is not installed"
    exit 1
fi

if [[ "$_java" ]]; then
    version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    if [[ "$version" < "1.7" ]]; then
        echo Require a java version not lower than 1.7
        exit 1
    fi
fi

# Check if mvn installed
MVN_INSTALL=$(which mvn 2>/dev/null | grep mvn | wc -l)
if [ $MVN_INSTALL -eq 0 ]; then
  echo "MVN is not installed. Exit"
  exit 1
fi

mvn clean package -DskipTests $*

BASEDIR=$(dirname "$0")
DIST_DIR=$BASEDIR/dist
BIN_DIR=$DIST_DIR/bin
LIB_DIR=$DIST_DIR/lib
VERSION=0.1.0-SNAPSHOT

if [ ! -d "$DIST_DIR" ]
then
  mkdir $DIST_DIR
else
  rm -r $DIST_DIR
  mkdir $DIST_DIR
fi

mkdir $BIN_DIR
mkdir $LIB_DIR

cp $BASEDIR/scripts/bigdl.sh $BIN_DIR/
cp $BASEDIR/scripts/classes.lst $BIN_DIR/
cp $BASEDIR/scripts/img_class.lst $BIN_DIR/

cp $BASEDIR/dl/target/bigdl-$VERSION-jar-with-dependencies.jar $LIB_DIR/
cp $BASEDIR/dl/target/bigdl-$VERSION-jar-with-dependencies-and-spark.jar $LIB_DIR/
cp $BASEDIR/dl/target/bigdl-$VERSION-python-api.zip $LIB_DIR/


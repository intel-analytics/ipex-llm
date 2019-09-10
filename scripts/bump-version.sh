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

#
# Use this script to bump bigdl source code version. It will search through the files listed in the
# python.version.lst, scala.version.lst(both in the scripts folder) and all pom.xml files, and
# replace the old version string with the new one.
# Usage: bump-version.sh [python|scala] old-version new-version
# The first argument specify which type of code you want to modify. The version formats of python
# and scala are different.
#
# The script will replace the old-version with new-version. The python version format is x.x.x for
# a release version number and x.x.x.devx for a developing version number. The x here is a digit
# number. The scala version format are similar but x.x.x-SNAPSHOT for a developing version number.
#

set -e

BASEDIR=$(dirname "$0")

update_file()
{
    FILE=$1
    FROM=$2
    TO=$3
    echo "Update $FILE"
    TMP="/tmp/bump-verison-bigdl.tmp"
    sed "s/$FROM/$TO/" $FILE>$TMP
    mv $TMP $FILE
}

verify_python_version()
{
    VERSION=$1
    if ! [[ $VERSION =~ ^[0-9]\.[0-9]+\.[0-9]$ ]] && ! [[ $VERSION =~ ^[0-9]\.[0-9]+\.[0-9]\.dev[0-9]$ ]]; then
        echo "Invalid python version $VERSION. It should be x.x.x or x.x.x.devx"
        exit 1
    fi
}

verify_scala_version()
{
    VERSION=$1
    if ! [[ $VERSION =~ ^[0-9]\.[0-9]+\.[0-9]$ ]] && ! [[ $VERSION =~ ^[0-9]\.[0-9]+\.[0-9]-SNAPSHOT$ ]]; then
        echo "Invalid scala version $VERSION. It should be x.x.x or x.x.x-SNAPSHOT"
        exit 1
    fi
}

# Script execution start from here
if [ "$#" -ne 3 ]; then
    echo "Invalid parameter number. Usage: bump-version.sh [python|scala] from_version to_version"
    exit 1
fi

if [ "$1" = "python" ]; then
    verify_python_version $2
    verify_python_version $3
    echo "Bump python version from $2 to $3..."
    while read p; do
        update_file "$BASEDIR/../$p" $2 $3
    done <$BASEDIR/python.version.lst
    echo done
elif [ "$1" = "scala" ]; then
    verify_scala_version $2
    verify_scala_version $3
    echo "Bump scala version from $2 to $3..."
    while read p; do
        update_file "$BASEDIR/../$p" $2 $3
    done <$BASEDIR/scala.version.lst

    # Find all pom.xml files and update
    for p in $(find $BASEDIR/../ -name pom.xml); do
        update_file "$BASEDIR/../$p" $2 $3
    done
    echo done
else
    echo "First argument must be either python or scala"
    exit 1
fi


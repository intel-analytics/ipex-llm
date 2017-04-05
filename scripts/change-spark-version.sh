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

set -e

VALID_VERSIONS=( 1.5p 2.0 )

usage() {
  echo "Usage: $(basename $0) [-h|--help] <version>
where :
  -h| --help Display this help text
  valid version values : ${VALID_VERSIONS[*]}
" 1>&2
  exit 1
}

if [[ ($# -ne 1) || ( $1 == "--help") ||  $1 == "-h" ]]; then
  usage
fi

TO_VERSION=$1

check_spark_version() {
  for i in ${VALID_VERSIONS[*]}; do [ $i = "$1" ] && return 0; done
  echo "Invalid Spark version: $1. Valid versions: ${VALID_VERSIONS[*]}" 1>&2
  exit 1
}

check_spark_version "$TO_VERSION"

if [ $TO_VERSION = "2.0" ]; then
  FROM_VERSION=""
  TO_VERSION="-SPARK_2.0"
else
  FROM_VERSION="-SPARK_2.0"
  TO_VERSION=""
fi

sed_i() {
  sed -e "$1" "$2" > "$2.tmp" && mv "$2.tmp" "$2"
}

export -f sed_i

BASEDIR=$(dirname $0)/../spark/dl
find "$BASEDIR" -name 'pom.xml' -not -path '*target*' -print \
  -exec bash -c "sed_i 's/\(^    <artifactId.*\)bigdl'$FROM_VERSION'\([^-parent]\)/\1bigdl'$TO_VERSION'\2/g' {}" \;

# Also update <spark.version> in parent POM
# Match any spark version to ensure independency
# sed_i '1,/<spark\.version>[0-9]*\.[0-9]*</s/<spark\.version>[0-9]*\.[0-9]*</<spark.version>'$TO_VERSION'</' \

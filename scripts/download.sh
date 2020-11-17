#!/bin/sh

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

HDFS_HOST=$1                                                                                               
set -e

BIGDL_VERSION=0.13.0-SNAPSHOT

if [ -d "images" ]
then
    rm -r images
fi
mkdir -p images/train
cd images/train
seq 1 1000 | xargs mkdir
cd ../../
export k=1
cat flickr.urls |  while read URL;
do
    wget --tries=1 ${URL} -P ./images/train/${k} -T 10;
    export k=$(($k + 1))
    if [ $k = '1001' ]; then
        export k=1
    fi
done
cp -r images/train/ images/val
mvn dependency:get -Dclassifier=jar-with-dependencies-and-spark -DrepoUrl=https://oss.sonatype.org/content/groups/public/ -DartifactId=bigdl -DgroupId=com.intel.analytics.bigdl -Dversion=$BIGDL_VERSION
if [ -d "seq" ]
then
    rm -r seq
fi
mkdir seq
export CORE_NUMBER=`cat /proc/cpuinfo | grep processor | wc -l`
java -cp ~/.m2/repository/com/intel/analytics/bigdl/bigdl/$BIGDL_VERSION/bigdl-$BIGDL_VERSION-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f ./images -o ./seq -p $CORE_NUMBER
rm seq/train/.*.crc
rm seq/val/.*.crc

HADOOP_HOME=$HOME/hadoop-2.7.3
if [ ! -d "hadoop-2.7.3" ]; then
        wget http://apache.claz.org/hadoop/common/hadoop-2.7.3/hadoop-2.7.3.tar.gz
        tar -xzf hadoop-2.7.3.tar.gz

cat << EOF > $HADOOP_HOME/etc/hadoop/core-site.xml
<configuration>
        <property>
                <name>fs.defaultFS</name>
                <value>hdfs://$HDFS_HOST:9000</value>
        </property>
</configuration>
EOF

fi

$HADOOP_HOME/bin/hadoop fs -mkdir -p /seq/
$HADOOP_HOME/bin/hadoop fs -copyFromLocal seq/* /seq/

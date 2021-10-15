#!/usr/bin/env bash

if [ ! -z "$1" ]
then
   DIR=$1
   cd "$DIR"
else
   DIR=$(dirname "$0")
   echo "Download path: $DIR"
   cd "$DIR"
fi

echo "Downloading machine_usage"
wget http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz

echo "Unzip machine_usage"
tar -zxvf machine_usage.tar.gz

echo "Finished"

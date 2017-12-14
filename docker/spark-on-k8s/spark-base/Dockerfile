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

#FROM openjdk:8-alpine

FROM ubuntu:14.04

ENV JAVA_HOME	/opt/jdk
ENV PATH	${JAVA_HOME}/bin:${PATH}

# If this docker file is being used in the context of building your images from a Spark distribution, the docker build
# command should be invoked from the top level directory of the Spark distribution. E.g.:
# docker build -t spark-base:latest -f dockerfiles/spark-base/Dockerfile .

RUN apt-get update && \
    apt-get install -y vim curl nano wget && \
#tini
    curl -fsSL https://github.com/krallin/tini/releases/download/v0.16.1/tini-static-amd64 -o /sbin/tini && \
    chmod +x /sbin/tini && \
#java
    wget --no-cookies --no-check-certificate --header "Cookie: gpw_e24=http%3A%2F%2Fwww.oracle.com%2F; oraclelicense=accept-securebackup-cookie" "http://download.oracle.com/otn-pub/java/jdk/8u152-b16/aa0333dd3019491ca4f6ddbe78cdb6d0/jdk-8u152-linux-x64.tar.gz" && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 /opt/jdk

#RUN apk upgrade --no-cache && \
#    apk add --no-cache bash tini && \
RUN mkdir -p /opt/spark && \
    mkdir -p /opt/spark/work-dir \
    touch /opt/spark/RELEASE && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd

COPY jars /opt/spark/jars
COPY bin /opt/spark/bin
COPY sbin /opt/spark/sbin
COPY conf /opt/spark/conf
COPY dockerfiles/spark-base/entrypoint.sh /opt/

ENV SPARK_HOME /opt/spark

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

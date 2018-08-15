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

ARG BIGDL_VERSION=0.6.0
ARG SPARK_VERSION=2.3.1

ENV JAVA_HOME	            /opt/jdk
ENV PATH	                ${JAVA_HOME}/bin:${PATH}
ENV SPARK_VERSION_ENV		${SPARK_VERSION}
ENV SPARK_HOME              /opt/spark
ENV BIGDL_VERSION_ENV		${BIGDL_VERSION}
ENV BIGDL_HOME			    /opt/bigdl-${BIGDL_VERSION}


# If this docker file is being used in the context of building your images from a Spark distribution, the docker build
# command should be invoked from the top level directory of the Spark distribution. E.g.:
# docker build -t spark-base:latest -f dockerfiles/spark-base/Dockerfile .

RUN apt-get update && \
    apt-get install -y vim curl nano wget unzip
#tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN chmod a+x /sbin/tini
#java
RUN wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 /opt/jdk

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
COPY kubernetes/dockerfiles/bigdl/entrypoint.sh /opt/


#python
ADD examples /opt/spark/examples
ADD python /opt/spark/python
RUN apt-get install -y software-properties-common python-software-properties python-pkg-resources && \
    add-apt-repository -y ppa:jonathonf/python-2.7 && \
    apt-get update && \
    apt-get install -y build-essential python python-setuptools python-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    pip install numpy six

ENV PYTHON_VERSION 2.7.13
ENV PYSPARK_PYTHON python
ENV PYSPARK_DRIVER_PYTHON python
ENV PYTHONPATH ${SPARK_HOME}/python/:${SPARK_HOME}/python/lib/py4j-0.10.4-src.zip:${PYTHONPATH}

#bigdl
ADD kubernetes/dockerfiles/bigdl/download-bigdl.sh /opt/
RUN chmod a+x /opt/download-bigdl.sh
RUN /opt/download-bigdl.sh

COPY kubernetes/dockerfiles/bigdl/mnist /tmp/

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

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


RUN apt-get update && \
    apt-get install -y vim curl nano wget unzip
#tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN chmod a+x /sbin/tini

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
COPY kubernetes/dockerfiles/spark2.3-k8s/entrypoint.sh /opt/

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
ENV PYTHONPATH ${SPARK_HOME}/python/:${SPARK_HOME}/python/lib/py4j-0.10.6-src.zip:${PYTHONPATH}

#java
RUN add-apt-repository ppa:openjdk-r/ppa -y && \
    apt-get update && \
    apt-get install openjdk-8-jdk -y && \
    update-alternatives --config java && \
    update-alternatives --config javac && \
    ln -s /usr/lib/jvm/java-8-openjdk-amd64/jre /opt/jdk

#bigdl
ADD kubernetes/dockerfiles/spark2.3-k8s/download-bigdl.sh /opt/
RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/download-bigdl.sh
RUN /opt/download-bigdl.sh

#COPY kubernetes/dockerfiles/spark2.3-k8s/mnist /tmp/

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

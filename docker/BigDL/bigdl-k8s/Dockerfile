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
    apt-get install -y vim curl nano wget
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
COPY dockerfiles/bigdl/entrypoint.sh /opt/


#python
ADD examples /opt/spark/examples
ADD python /opt/spark/python
RUN apt-get install -y build-essential python python-setuptools python-dev && \
    easy_install pip && \
    pip install numpy scipy pandas scikit-learn matplotlib seaborn jupyter wordcloud

ENV PYTHON_VERSION 2.7.13
ENV PYSPARK_PYTHON python
ENV PYSPARK_DRIVER_PYTHON python
ENV PYTHONPATH ${SPARK_HOME}/python/:${SPARK_HOME}/python/lib/py4j-0.10.4-src.zip:${PYTHONPATH}

CMD SPARK_CLASSPATH="${SPARK_HOME}/jars/*" && \
    env | grep SPARK_JAVA_OPT_ | sed 's/[^=]*=\(.*\)/\1/g' > /tmp/java_opts.txt && \
    readarray -t SPARK_DRIVER_JAVA_OPTS < /tmp/java_opts.txt && \
    if ! [ -z ${SPARK_MOUNTED_CLASSPATH+x} ]; then SPARK_CLASSPATH="$SPARK_MOUNTED_CLASSPATH:$SPARK_CLASSPATH"; fi && \
    if ! [ -z ${SPARK_SUBMIT_EXTRA_CLASSPATH+x} ]; then SPARK_CLASSPATH="$SPARK_SUBMIT_EXTRA_CLASSPATH:$SPARK_CLASSPATH"; fi && \
    if ! [ -z ${SPARK_EXTRA_CLASSPATH+x} ]; then SPARK_CLASSPATH="$SPARK_EXTRA_CLASSPATH:$SPARK_CLASSPATH"; fi && \
    if ! [ -z ${SPARK_MOUNTED_FILES_DIR+x} ]; then cp -R "$SPARK_MOUNTED_FILES_DIR/." .; fi && \
    if ! [ -z ${SPARK_MOUNTED_FILES_FROM_SECRET_DIR+x} ]; then cp -R "$SPARK_MOUNTED_FILES_FROM_SECRET_DIR/." .; fi && \
    ${JAVA_HOME}/bin/java "${SPARK_DRIVER_JAVA_OPTS[@]}" -cp $SPARK_CLASSPATH -Xms$SPARK_DRIVER_MEMORY -Xmx$SPARK_DRIVER_MEMORY -Dspark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS $SPARK_DRIVER_CLASS $PYSPARK_PRIMARY $PYSPARK_FILES $SPARK_DRIVER_ARGS

#bigdl
ADD ./download-bigdl.sh /opt/
RUN chmod a+x /opt/download-bigdl.sh
RUN /opt/download-bigdl.sh

COPY kubernetes/bigdl/mnist /tmp/

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

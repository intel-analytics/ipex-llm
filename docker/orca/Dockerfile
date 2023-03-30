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

ARG SPARK_VERSION=2.4.6
ARG PY4J_VERSION=0.10.7
ARG SPARK_HOME=/opt/spark
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG DEBIAN_FRONTEND=noninteractive

# stage.1 jdk & spark
FROM ubuntu:20.04 as spark
ARG SPARK_VERSION
ARG PY4J_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_HOME
ARG DEBIAN_FRONTEND
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          ${SPARK_HOME}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip git && \
# java
    wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk && \
# spark
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    cp /opt/spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt && \
    # remove log4j 1.x jars
    rm -f ${SPARK_HOME}/jars/log4j-1.2.17.jar && \
    rm -f ${SPARK_HOME}/jars/slf4j-log4j12-1.7.16.jar && \
    rm -f ${SPARK_HOME}/jars/apache-log4j-extras-1.2.17.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/slf4j/slf4j-reload4j/1.7.35/slf4j-reload4j-1.7.35.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar

ADD ./log4j2.xml ${SPARK_HOME}/conf/log4j2.xml
ADD ./spark-defaults.conf ${SPARK_HOME}/conf/spark-defaults.conf
RUN ln -fs /bin/bash /bin/sh
RUN if [ $SPARK_VERSION = "3.1.3" ]; then \
        rm $SPARK_HOME/jars/okhttp-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar; \
    elif [ $SPARK_VERSION = "2.4.6" ]; then \
        rm $SPARK_HOME/jars/kubernetes-client-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/io/fabric8/kubernetes-client/4.4.2/kubernetes-client-4.4.2.jar; \
    fi

# stage.2 bigdl
FROM ubuntu:20.04 as bigdl
ARG SPARK_VERSION
ARG BIGDL_VERSION
ARG DEBIAN_FRONTEND
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /opt/bigdl-${BIGDL_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip  git
ADD ./download-bigdl.sh /opt

RUN chmod a+x /opt/download-bigdl.sh && \
    mkdir -p /opt/bigdl-examples/python
RUN /opt/download-bigdl.sh && \
    rm bigdl*.zip

# stage.3 bigdl-tutorials
FROM ubuntu:20.04
ARG BIGDL_VERSION
ARG SPARK_VERSION
ARG RUNTIME_SPARK_MASTER=local[4]
ARG RUNTIME_DRIVER_CORES=4
ARG RUNTIME_DRIVER_MEMORY=20g
ARG RUNTIME_EXECUTOR_CORES=4
ARG RUNTIME_EXECUTOR_MEMORY=20g
ARG RUNTIME_EXECUTOR_INSTANCES=1
ARG DEBIAN_FRONTEND
ENV SPARK_VERSION                   ${SPARK_VERSION}
ENV BIGDL_VERSION                   ${BIGDL_VERSION}
ENV RUNTIME_SPARK_MASTER            ${RUNTIME_SPARK_MASTER}
ENV RUNTIME_DRIVER_CORES            ${RUNTIME_DRIVER_CORES}
ENV RUNTIME_DRIVER_MEMORY           ${RUNTIME_DRIVER_MEMORY}
ENV RUNTIME_EXECUTOR_CORES          ${RUNTIME_EXECUTOR_CORES}
ENV RUNTIME_EXECUTOR_MEMORY         ${RUNTIME_EXECUTOR_MEMORY}
ENV RUNTIME_EXECUTOR_INSTANCES      ${RUNTIME_EXECUTOR_INSTANCES}
ENV SPARK_HOME                      /opt/work/spark-${SPARK_VERSION}
ENV BIGDL_HOME                      /opt/work/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME                       /opt/jdk
ENV BIGDL_CLASSPATH                 ${BIGDL_HOME}/jars/*
ENV PYTHONPATH                      ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip:${BIGDL_HOME}/conf/spark-bigdl.conf:${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-${PY4J_VERSION}-src.zip
ENV PATH                            ${JAVA_HOME}/bin:${PATH}

WORKDIR /opt/work

COPY --from=spark /opt/jdk /opt/jdk
COPY --from=spark /opt/spark /opt/work/spark-${SPARK_VERSION}
COPY --from=bigdl /opt/bigdl-${BIGDL_VERSION} /opt/work/bigdl-${BIGDL_VERSION}

ADD ./install-orca-env.sh /opt
RUN chmod a+x /opt/install-orca-env.sh

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip git && \
    apt-get install -y gcc g++ make && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd && \
# Install Miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    chmod +x Miniconda3-4.5.4-Linux-x86_64.sh && \
    ./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local && \
    rm Miniconda3-4.5.4-Linux-x86_64.sh && \
# python
    /opt/install-orca-env.sh

RUN echo "source activate bigdl" > ~/.bashrc
ENV PATH /usr/local/envs/bigdl/bin:$PATH


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
ARG HADOOP_VERSION=2.7
ARG BIGDL_VERSION=2.5.0-SNAPSHOT

# stage.1 spark
FROM ubuntu:20.04 as spark
ARG SPARK_VERSION
ARG HADOOP_VERSION
ARG SPARK_HOME=/opt/spark

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends apt-utils wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --progress=dot:giga --no-check-certificate https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    # remove log4j 1.x jars
    rm -f ${SPARK_HOME}/jars/log4j-1.2.17.jar && \
    rm -f ${SPARK_HOME}/jars/slf4j-log4j12-1.7.16.jar && \
    rm -f ${SPARK_HOME}/jars/apache-log4j-extras-1.2.17.jar && \
    wget --progress=dot:giga --no-check-certificate -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar && \
    wget --progress=dot:giga --no-check-certificate -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/slf4j/slf4j-reload4j/1.7.35/slf4j-reload4j-1.7.35.jar && \
    wget --progress=dot:giga --no-check-certificate -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget --progress=dot:giga --no-check-certificate -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget --progress=dot:giga --no-check-certificate -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar && \
    # remove conflict jars
    rm -f ${SPARK_HOME}/jars/guava* && \
    rm -f ${SPARK_HOME}/jars/metrics-core* && \
    rm -f ${SPARK_HOME}/jars/jackson-annotations* ${SPARK_HOME}/jars/jackson-core-2.* ${SPARK_HOME}/jars/jackson-databind* ${SPARK_HOME}/jars/jackson-module-scala* && \
    rm -f ${SPARK_HOME}/jars/hk2-* ${SPARK_HOME}/jars/jersey-client* ${SPARK_HOME}/jars/jersey-server* ${SPARK_HOME}/jars/jersey-common*

# stage.2 friesian-serving-docker
FROM ubuntu:20.04

ARG SPARK_VERSION=2.4.6
ARG BIGDL_VERSION=2.5.0-SNAPSHOT

ENV SPARK_HOME                      /opt/work/spark-${SPARK_VERSION}
ENV LD_LIBRARY_PATH /opt/work/lib
ENV SERVING_JAR_PATH /opt/work/bigdl-friesian-serving.jar

WORKDIR /opt/work

COPY --from=spark /opt/spark ${SPARK_HOME}

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends openjdk-8-jdk libgomp1 vim net-tools openssh-server wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir logs

COPY download-friesian-serving.sh /opt/work

RUN chmod a+x download-friesian-serving.sh && \
    /opt/work/download-friesian-serving.sh

COPY lib/* /opt/work/lib/
COPY start_service.sh /opt/work/
COPY jersey/* /opt/work/jersey/
RUN chmod a+x /opt/work/start_service.sh

ENTRYPOINT ["/opt/work/start_service.sh"]

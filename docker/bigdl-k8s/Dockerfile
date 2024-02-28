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
ARG SPARK_HOME=/opt/spark
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ARG BIGDL_VERSION=2.5.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG DEBIAN_FRONTEND=noninteractive

# stage.1 jdk & spark
FROM ubuntu:20.04 as spark
ARG SPARK_VERSION
ARG HADOOP_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_HOME
ARG DEBIAN_FRONTEND
ARG TINI_VERSION
ENV TINI_VERSION                        ${TINI_VERSION}
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          ${SPARK_HOME}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip git && \
# java
    wget -O jdk.tar.gz $JDK_URL && \
    gunzip jdk.tar.gz && \
    mkdir /opt/jdk$JDK_VERSION && \
    tar -xf jdk.tar -C /opt/jdk$JDK_VERSION --strip-components 1 && \
    rm jdk.tar && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk && \
# spark
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
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
    

COPY ./log4j2.xml ${SPARK_HOME}/conf/log4j2.xml
RUN ln -fs /bin/bash /bin/sh
RUN if [ $SPARK_VERSION = "3.4.1" ]; then \
        rm $SPARK_HOME/jars/okhttp-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar; \
    elif [ $SPARK_VERSION = "2.4.6" ]; then \
        rm $SPARK_HOME/jars/kubernetes-client-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/io/fabric8/kubernetes-client/4.4.2/kubernetes-client-4.4.2.jar; \
    fi

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN git clone https://github.com/tensorflow/models/ /opt/models

# stage.2 bigdl
FROM ubuntu:20.04 as bigdl
ARG SPARK_VERSION
ARG BIGDL_VERSION
ARG DEBIAN_FRONTEND
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /opt/bigdl-${BIGDL_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip git
COPY ./download-bigdl.sh /opt

RUN chmod a+x /opt/download-bigdl.sh && \
    mkdir -p /opt/bigdl-examples/python
RUN /opt/download-bigdl.sh && \
    rm bigdl*.zip

# stage.3 copies layer
FROM ubuntu:20.04 as copies-layer
ARG BIGDL_VERSION
ARG DEBIAN_FRONTEND

COPY --from=bigdl /opt/bigdl-${BIGDL_VERSION} /opt/bigdl-${BIGDL_VERSION}
COPY --from=spark /opt/jdk /opt/jdk
COPY --from=spark /opt/spark /opt/spark
COPY --from=spark /opt/spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt


# stage.4
FROM ubuntu:20.04
ARG BIGDL_VERSION
ARG SPARK_VERSION
ARG SPARK_HOME
ARG TINI_VERSION
ARG DEBIAN_FRONTEND

ENV BIGDL_HOME                          /opt/bigdl-${BIGDL_VERSION}
ENV BIGDL_VERSION                       ${BIGDL_VERSION}
ENV SPARK_HOME                          ${SPARK_HOME}
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV OMP_NUM_THREADS                     4
ENV RUNTIME_SPARK_MASTER                local[4]
ENV RUNTIME_K8S_SERVICE_ACCOUNT         spark
ENV RUNTIME_K8S_SPARK_IMAGE             intelanalytics/bigdl-k8s:${BIGDL_VERSION}-${SPARK_VERSION}
ENV RUNTIME_DRIVER_HOST                 localhost
ENV RUNTIME_DRIVER_PORT                 54321
ENV RUNTIME_EXECUTOR_CORES              4
ENV RUNTIME_EXECUTOR_MEMORY             20g
ENV RUNTIME_EXECUTOR_INSTANCES          1
ENV RUNTIME_TOTAL_EXECUTOR_CORES        4
ENV RUNTIME_DRIVER_CORES                4
ENV RUNTIME_DRIVER_MEMORY               10g
ENV RUNTIME_PERSISTENT_VOLUME_CLAIM     myvolumeclaim
ENV JAVA_HOME                           /opt/jdk
ENV PATH ${BIGDL_HOME}/bin/cluster-serving:${JAVA_HOME}/bin:${PATH}
ENV TINI_VERSION                        ${TINI_VERSION}
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8


COPY --from=copies-layer /opt /opt
COPY --from=spark /sbin/tini /sbin/tini

RUN mkdir -p /opt/bigdl-examples/python && \
    mkdir -p /opt/bigdl-examples/scala && \
    apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl libgomp1 nano wget unzip git && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 

RUN chmod +x /sbin/tini
RUN cp /sbin/tini /usr/bin/tini

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

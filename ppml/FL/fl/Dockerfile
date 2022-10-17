ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.2.0-SNAPSHOT

ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=8u192
ARG JDK_URL
ARG SPARK_JAR_REPO_URL

# Stage.1 JDK & Scala & Spark
FROM ubuntu:20.04 as spark
ARG SPARK_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_JAR_REPO_URL
ENV SPARK_VERSION       ${SPARK_VERSION}
ENV JAVA_HOME           /opt/jdk$JDK_VERSION
ENV PATH                ${JAVA_HOME}/bin:${PATH}
RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-utils wget unzip patch zip git maven nasm
# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk
# scala
RUN cd / && wget -c https://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.tgz && \
    (cd / && gunzip < scala-2.11.8.tgz)|(cd /opt && tar -xvf -) && \
    rm /scala-2.11.8.tgz

# spark
RUN cd /opt && \
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    cp spark-${SPARK_VERSION}/conf/log4j.properties.template spark-${SPARK_VERSION}/conf/log4j.properties && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> spark-${SPARK_VERSION}/conf/log4j.properties \
    rm spark-${SPARK_VERSION}/python/lib/pyspark.zip && \
    rm spark-${SPARK_VERSION}/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/spark-launcher_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/spark-network-common_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/hadoop-common-3.2.0.jar && \
    rm spark-${SPARK_VERSION}/jars/hive-exec-2.3.7-core.jar
ADD ./log4j2.xml /opt/spark-${SPARK_VERSION}/conf/log4j2.xml
# spark modification
RUN cd /opt && \
    wget $SPARK_JAR_REPO_URL/spark-core_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-network-common_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-examples_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-launcher_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/pyspark.zip && \
    mv /opt/spark-core_2.12-$SPARK_VERSION.jar  /opt/spark-${SPARK_VERSION}/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-launcher_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/jars/spark-launcher_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-kubernetes_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-network-common_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/jars/spark-network-common_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-examples_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar && \
    mv /opt/pyspark.zip /opt/spark-${SPARK_VERSION}/python/lib/pyspark.zip && \
    sed -i 's/\#\!\/usr\/bin\/env bash/\#\!\/usr\/bin\/env bash\nset \-x/' /opt/spark-${SPARK_VERSION}/bin/spark-class && \
    rm -f /opt/spark-${SPARK_VERSION}/jars/log4j-1.2.17.jar && \
    rm -f /opt/spark-${SPARK_VERSION}/jars/slf4j-log4j12-1.7.16.jar && \
    rm -f /opt/spark-${SPARK_VERSION}/jars/apache-log4j-extras-1.2.17.jar && \
    rm -r /opt/spark-${SPARK_VERSION}/jars/slf4j-log4j12-1.7.30.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/slf4j/slf4j-reload4j/1.7.35/slf4j-reload4j-1.7.35.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/wildfly/openssl/wildfly-openssl/1.0.7.Final/wildfly-openssl-1.0.7.Final.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure/3.2.0/hadoop-azure-3.2.0.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure-datalake/3.2.0/hadoop-azure-datalake-3.2.0.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/com/microsoft/azure/azure-storage/7.0.0/azure-storage-7.0.0.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/com/microsoft/azure/azure-data-lake-store-sdk/2.2.9/azure-data-lake-store-sdk-2.2.9.jar

# Stage.2 BigDL
FROM ubuntu:20.04 as bigdl
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /bigdl-${BIGDL_VERSION}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip git
RUN wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
    chmod a+x ./download-bigdl.sh
RUN ./download-bigdl.sh && \
    rm bigdl*.zip

# Stage.3 Federal Learning
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as fl
ARG BIGDL_VERSION
ARG SPARK_VERSION
ARG TINI_VERSION
ENV BIGDL_VERSION                       ${BIGDL_VERSION}
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          /ppml/work/spark-${SPARK_VERSION}
ENV BIGDL_HOME                          /ppml/work/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME                           /opt/jdk8
ENV PATH                                ${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP                            127.0.0.1
ENV SPARK_MASTER_IP                     127.0.0.1
ENV SPARK_MASTER_PORT                   7077
ENV SPARK_MASTER_WEBUI_PORT             8080
ENV SPARK_MASTER                        spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT
ENV SPARK_WORKER_PORT                   8082
ENV SPARK_WORKER_WEBUI_PORT             8081
ENV SPARK_DRIVER_PORT                   10027
ENV SPARK_DRIVER_BLOCK_MANAGER_PORT     10026
ENV SPARK_DRIVER_IP                     $LOCAL_IP
ENV SPARK_BLOCK_MANAGER_PORT            10025
ENV TINI_VERSION                        $TINI_VERSION
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8

RUN mkdir -p /ppml/work && \
    mkdir -p /ppml/work/lib && \
    mkdir -p /ppml/work/keys && \
    mkdir -p /ppml/work/password && \
    mkdir -p /ppml/work/data && \
    mkdir -p /ppml/work/models && \
    mkdir -p /ppml/work/apps && \
    mkdir -p /ppml/work/examples/bigdl && \
    mkdir -p /root/.keras/datasets && \
    mkdir -p /root/.zinc && \
    mkdir -p /root/.m2 && \
    mkdir -p /root/.kube/ && \
    mkdir -p /ppml/fl

COPY --from=bigdl /bigdl-${BIGDL_VERSION}/python/start-fl-server.py /ppml/fl
COPY --from=bigdl /bigdl-${BIGDL_VERSION}/python/start-fgboost-server.py /ppml/fl
COPY --from=spark /opt/jdk  /opt/jdk8
COPY --from=spark /opt/scala-2.11.8  /opt/scala-2.11.8
COPY --from=spark /opt/spark-${SPARK_VERSION} /ppml/work/spark-${SPARK_VERSION}
COPY --from=spark /opt/spark-${SPARK_VERSION}/examples/src/main/resources /ppml/examples/src/main/resources
COPY --from=bigdl /bigdl-${BIGDL_VERSION} ${BIGDL_HOME}

ADD ./start-scripts /ppml/work/start-scripts

# python
RUN rm -rf /usr/lib/python3/dist-packages/protobuf-3.6.1.egg-info && \
    rm -rf /usr/lib/python3/dist-packages/google/protobuf && \
    pip install --no-cache-dir numpy scipy && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir protobuf==3.19.0 tensorflow==1.15 && \
    pip install --no-cache-dir torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir py4j && \
    python3 -m pip install click && \
    ln -s /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /usr/lib/x86_64-linux-gnu/libtinfo.so.5 && \
    pip install --no-cache-dir pyyaml && \
    pip install --no-cache-dir markupsafe==2.0.1 && \
    chmod a+x /ppml/work/start-scripts/*

ADD ./entrypoint.sh /opt/entrypoint.sh
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN rm $SPARK_HOME/jars/okhttp-*.jar && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar && \
    wget -P $SPARK_HOME/jars https://github.com/xerial/sqlite-jdbc/releases/download/3.36.0.1/sqlite-jdbc-3.36.0.1.jar && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/verify-attestation-service.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs



ENV PYTHONPATH /usr/lib/python3.7:/usr/lib/python3.7/lib-dynload:/usr/local/lib/python3.7/dist-packages:/usr/lib/python3/dist-packages
WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]


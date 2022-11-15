ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.2.0-SNAPSHOT
ARG SPARK_VERSION=3.1.3
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=8u192
ARG JDK_URL
ARG SPARK_JAR_REPO_URL

# Stage.1 Spark & Hadoop & Hive
FROM ubuntu:20.04 as bigdata
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG SPARK_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_JAR_REPO_URL

ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV JAVA_HOME                           /opt/jdk${JDK_VERSION}
ENV PATH                                ${JAVA_HOME}/bin:${PATH}

RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-utils wget unzip patch zip git maven nasm
# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

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
# hadoop
RUN cd /opt && \
    apt-get update --fix-missing && \
    apt-get install -y build-essential && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.bz2 && \
    tar jxvf protobuf-2.5.0.tar.bz2 && \
    cd protobuf-2.5.0 && \
    ./configure && \
    make && \
    make check && \
    export LD_LIBRARY_PATH=/usr/local/lib && \
    make install && \
    protoc --version && \
    cd /opt && \
    git clone https://github.com/analytics-zoo/hadoop.git && \
    cd hadoop && \
    git checkout branch-3.2.0-ppml && \
    cd hadoop-common-project/hadoop-common && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    mvn -T 16 -DskipTests=true clean package && \
    mv /opt/hadoop/hadoop-common-project/hadoop-common/target/hadoop-common-3.2.0.jar /opt/spark-${SPARK_VERSION}/jars/hadoop-common-3.2.0.jar
# hive
RUN cd /opt && \
    git clone https://github.com/analytics-zoo/hive.git && \
    cd hive && \
    git checkout branch-2.3.7-ppml && \
    cd ql && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    mvn -T 16 -DskipTests=true clean package && \
    mv /opt/hive/ql/target/hive-exec-2.3.7-core.jar /opt/spark-${SPARK_VERSION}/jars/hive-exec-2.3.7-core.jar
RUN ls -al /opt

# stage.2 gramine
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

ARG SPARK_VERSION
ARG TINI_VERSION

ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          /ppml/spark-${SPARK_VERSION}
ENV LOCAL_IP                            127.0.0.1
ENV TINI_VERSION                        $TINI_VERSION
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8

RUN mkdir -p /ppml/lib && \
    mkdir -p /ppml/keys && \
    mkdir -p /ppml/password && \
    mkdir -p /ppml/data && \
    mkdir -p /ppml/models && \
    mkdir -p /ppml/apps

COPY --from=bigdata /opt/spark-${SPARK_VERSION} /ppml/spark-${SPARK_VERSION}
COPY --from=bigdata /opt/spark-${SPARK_VERSION}/examples/src/main/resources /ppml/examples/src/main/resources

ADD ./bigdl-ppml-submit.sh /ppml/bigdl-ppml-submit.sh
ADD ./spark-executor-template.yaml /ppml/spark-executor-template.yaml
ADD ./spark-driver-template.yaml /ppml/spark-driver-template.yaml
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN rm $SPARK_HOME/jars/okhttp-*.jar && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar && \
    wget -P $SPARK_HOME/jars https://github.com/xerial/sqlite-jdbc/releases/download/3.36.0.1/sqlite-jdbc-3.36.0.1.jar && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/bigdl-ppml-submit.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs


ENTRYPOINT [ "/opt/entrypoint.sh" ]
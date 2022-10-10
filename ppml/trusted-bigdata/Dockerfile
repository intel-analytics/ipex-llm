ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=8u192
ARG JDK_URL
ARG SPARK_JAR_REPO_URL

# Stage.1 JDK & Scala & Spark & Hadoop & Hive & Flink
FROM ubuntu:20.04 as spark
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG SPARK_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_JAR_REPO_URL
ARG FLINK_VERSION=1.11.3
ARG SCALA_VERSION=2.12
ARG SCALA_SUB_VERSION_NUM=10

ENV SPARK_VERSION       ${SPARK_VERSION}
ENV JAVA_HOME           /opt/jdk$JDK_VERSION
ENV PATH                ${JAVA_HOME}/bin:${PATH}
ENV FLINK_VERSION ${FLINK_VERSION}
ENV SCALA_VERSION ${SCALA_VERSION}
ENV SCALA_SUB_VERSION_NUM ${SCALA_SUB_VERSION_NUM}

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
RUN cd / && wget -c https://downloads.lightbend.com/scala/${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}/scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}.tgz && \
    (cd / && gunzip < scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}.tgz)|(cd /opt && tar -xvf -) && \
    rm /scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}.tgz

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
# flink
RUN cd /opt && \
    wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_${SCALA_VERSION}.tgz && \
    tar -zxvf flink-${FLINK_VERSION}-bin-scala_${SCALA_VERSION}.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_${SCALA_VERSION}.tgz && \
    rm /opt/flink-${FLINK_VERSION}/lib/log4j-1.2-api-2.12.1.jar && \
    rm /opt/flink-${FLINK_VERSION}/lib/log4j-api-2.12.1.jar && \
    rm /opt/flink-${FLINK_VERSION}/lib/log4j-core-2.12.1.jar && \
    rm /opt/flink-${FLINK_VERSION}/lib/log4j-slf4j-impl-2.12.1.jar && \
    wget -P /opt/flink-${FLINK_VERSION}/lib/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar && \
    wget -P /opt/flink-${FLINK_VERSION}/lib/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget -P /opt/flink-${FLINK_VERSION}/lib/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget -P /opt/flink-${FLINK_VERSION}/lib/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar
RUN ls -al /opt
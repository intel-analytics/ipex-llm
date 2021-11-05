FROM krallin/ubuntu-tini AS tini
FROM occlum/occlum:0.24.0-ubuntu18.04 AS occlum

ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2
ARG HADOOP_VERSION=3.2.0
ENV HADOOP_VERSION=${HADOOP_VERSION}
ENV SPARK_VERSION=${SPARK_VERSION}
ENV SPARK_HOME=/opt/spark
ENV BIGDL_VERSION=${BIGDL_VERSION}
ENV BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}

ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.bz2 && \
    tar jxvf protobuf-2.5.0.tar.bz2 && \
    cd protobuf-2.5.0 && \
    ./configure && \
    make && \
    make check && \
    export LD_LIBRARY_PATH=/usr/local/lib && \
    make install

RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 

COPY --from=tini /usr/local/bin/tini /sbin/tini

# scala
RUN cd / && wget -c https://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.tgz && \
    (cd / && gunzip < scala-2.11.8.tgz)|(cd /opt && tar -xvf -) && \
    rm /scala-2.11.8.tgz
# maven
RUN cd /opt && \
    wget https://archive.apache.org/dist/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz && \
    tar -zxvf apache-maven-3.6.3-bin.tar.gz
# spark
RUN cd /opt && \
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    cp spark-${SPARK_VERSION}/conf/log4j.properties.template spark-${SPARK_VERSION}/conf/log4j.properties && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> spark-${SPARK_VERSION}/conf/log4j.properties
# spark modification
# org.apache.spark.util.Utils to disable chmod fork
RUN cd /opt && \
    mkdir src && \
    cd src && \
    git clone https://github.com/apache/spark.git && \
    cd spark && \
    git checkout tags/v3.1.2 -b branch-3.1.2 && \
    wget https://raw.githubusercontent.com/intel-analytics/BigDL/branch-2.0/ppml/trusted-big-data-ml/python/docker-graphene/spark-3.1.2.patch && \
    git apply /opt/src/spark/spark-3.1.2.patch && \
    git status

RUN cd /opt/src/spark && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean install -pl core && \
    cd resource-managers/kubernetes/core && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean package
# hadoop
# org.apache.hadoop.util.Shell.java to disable setsid
RUN cd /opt/src && \
    wget https://raw.githubusercontent.com/intel-analytics/BigDL/branch-2.0/ppml/trusted-big-data-ml/python/docker-graphene/hadoop-common-shell.patch && \
    git clone https://github.com/apache/hadoop.git && \
    cd hadoop && \
    git checkout rel/release-${HADOOP_VERSION} -b branch-${HADOOP_VERSION} && \
    cd hadoop-common-project/hadoop-common && \
    patch src/main/java/org/apache/hadoop/util/Shell.java /opt/src/hadoop-common-shell.patch && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean package && \
    ls /opt/hadoop/hadoop-common-project/hadoop-common/target/hadoop-common-${HADOOP_VERSION}.jar

# Remove fork with libhadoop.so and spark-network-common.jar
RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so -P /lib/ && \
    rm -f /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar && \
    wget -O /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.12-${SPARK_VERSION}.jar

RUN rm -f ${SPRK_HOME}/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    rm -f ${SPRK_HOME}/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    rm -f ${SPRK_HOME}/jars/hadoop-common-${HADOOP_VERSION}.jar && \
    cp -f /opt/src/spark/resource-managers/kubernetes/core/target/spark-kubernetes_2.12-$SPARK_VERSION.jar ${SPRK_HOME}/jars && \
    cp -f /opt/src/spark/core/target/spark-core_2.12-$SPARK_VERSION.jar ${SPRK_HOME}/jars && \
    cp -f /opt/src/hadoop/hadoop-common-project/hadoop-common/target/hadoop-common-${HADOOP_VERSION}.jar ${SPRK_HOME}/jars && \
    rm -rf /opt/src

COPY ./entrypoint.sh /opt/
COPY ./init.sh /opt/

RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/init.sh

WORKDIR /opt/

ENTRYPOINT [ "/opt/entrypoint.sh" ]

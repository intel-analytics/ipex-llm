FROM krallin/ubuntu-tini AS tini

FROM ubuntu:20.04 as bigdl

ARG SPARK_VERSION=3.1.2
ARG HADOOP_VERSION=3.2.0
ENV HADOOP_VERSION=${HADOOP_VERSION}
ENV SPARK_VERSION=${SPARK_VERSION}
ENV SPARK_HOME=/opt/spark

ARG SPARK_JAR_REPO_URL

ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ENV HTTP_PROXY=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT
ENV HTTPS_PROXY=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk build-essential wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# maven
RUN cd /opt && \
    wget https://archive.apache.org/dist/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz && \
    tar -zxvf apache-maven-3.6.3-bin.tar.gz

# spark
# TODO change to build from source instead of download from spark binary
RUN cd /opt && \
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    cp spark/conf/log4j.properties.template spark/conf/log4j.properties && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> spark/conf/log4j.properties && \
    rm spark/python/lib/pyspark.zip && \
    rm spark/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    rm spark/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    rm spark/jars/spark-network-common_2.12-$SPARK_VERSION.jar && \
    rm spark/jars/hadoop-common-3.2.0.jar && \
    rm spark/jars/hive-exec-2.3.7-core.jar

# spark modification and download test jars
RUN cd /opt && \
    wget $SPARK_JAR_REPO_URL/spark-core_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-network-common_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/pyspark.zip && \
    mv /opt/spark-core_2.12-$SPARK_VERSION.jar  /opt/spark/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-kubernetes_2.12-$SPARK_VERSION.jar /opt/spark/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-network-common_2.12-$SPARK_VERSION.jar /opt/spark/jars/spark-network-common_2.12-$SPARK_VERSION.jar && \
    mv /opt/pyspark.zip /opt/spark/python/lib/pyspark.zip && \
    mkdir /opt/spark/test-jars && \
    cd /opt/spark/test-jars && \
    wget $SPARK_JAR_REPO_URL/spark-core_2.12-$SPARK_VERSION-tests.jar && \
    wget $SPARK_JAR_REPO_URL/spark-catalyst_2.12-$SPARK_VERSION-tests.jar && \
    wget $SPARK_JAR_REPO_URL/spark-sql_2.12-$SPARK_VERSION-tests.jar && \
    wget https://repo1.maven.org/maven2/org/scalactic/scalactic_2.12/3.1.4/scalactic_2.12-3.1.4.jar && \
    wget https://repo1.maven.org/maven2/org/scalatest/scalatest_2.12/3.1.4/scalatest_2.12-3.1.4.jar && \
    wget https://repo1.maven.org/maven2/org/mockito/mockito-core/3.4.6/mockito-core-3.4.6.jar && \
    wget https://repo1.maven.org/maven2/com/h2database/h2/1.4.195/h2-1.4.195.jar && \
    wget https://repo1.maven.org/maven2/com/ibm/db2/jcc/11.5.0.0/jcc-11.5.0.0.jar && \
    wget https://repo1.maven.org/maven2/org/apache/parquet/parquet-avro/1.10.1/parquet-avro-1.10.1.jar && \
    wget https://repo1.maven.org/maven2/net/bytebuddy/byte-buddy/1.10.13/byte-buddy-1.10.13.jar && \
    wget https://repo1.maven.org/maven2/org/postgresql/postgresql/42.2.6/postgresql-42.2.6.jar && \
    wget https://repo1.maven.org/maven2/org/scalatestplus/scalatestplus-mockito_2.12/1.0.0-SNAP5/scalatestplus-mockito_2.12-1.0.0-SNAP5.jar && \
    wget https://repo1.maven.org/maven2/org/scalatestplus/scalatestplus-scalacheck_2.12/3.1.0.0-RC2/scalatestplus-scalacheck_2.12-3.1.0.0-RC2.jar
    

RUN mkdir -p /opt/src

# hadoop
RUN cd /opt/src && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.bz2 && \
    tar jxvf protobuf-2.5.0.tar.bz2 && \
    cd protobuf-2.5.0 && \
    ./configure && \
    make && \
    make check && \
    export LD_LIBRARY_PATH=/usr/local/lib && \
    make install && \
    rm -f protobuf-2.5.0.tar.bz2 && \
    cd /opt/src && \
    git clone https://github.com/analytics-zoo/hadoop.git && \
    cd hadoop && \
    git checkout branch-3.2.0-ppml && \
    cd hadoop-common-project/hadoop-common && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean package

# hive
RUN cd /opt/src && \
    git clone https://github.com/analytics-zoo/hive.git && \
    cd hive && \
    git checkout branch-2.3.7-ppml && \
    cd ql && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean package && \
    mv /opt/src/hive/ql/target/hive-exec-2.3.7-core.jar /opt/spark/jars/hive-exec-2.3.7-core.jar

# Remove fork with libhadoop.so and spark-network-common.jar
RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so -P /opt/ && \
    cp -f /opt/src/hadoop/hadoop-common-project/hadoop-common/target/hadoop-common-${HADOOP_VERSION}.jar ${SPARK_HOME}/jars && \
    rm -rf /opt/src

# sbt
RUN apt-get update && \
    apt-get install apt-transport-https curl gnupg -yqq && \
    echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list && \
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list && \
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/scalasbt-release.gpg --import && \
    chmod 644 /etc/apt/trusted.gpg.d/scalasbt-release.gpg && \
    apt-get update && \
    apt-get install sbt

# Prepare tpch 
## 2.11.7 2.12.1 for scala version
## 2.4.0 3.1.2 for spark version
RUN cd /opt && \
    git clone https://github.com/intel-analytics/zoo-tutorials.git && \
    cd zoo-tutorials/tpch-spark && \
    sed -i 's/2.11.7/2.12.1/g' tpch.sbt && \
    sed -i 's/2.4.0/3.1.2/g' tpch.sbt && \
    sbt package

FROM occlum/occlum:0.26.4-ubuntu20.04 as ppml

ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2
ARG HADOOP_VERSION=3.2.0
ENV HADOOP_VERSION=${HADOOP_VERSION}
ENV SPARK_VERSION=${SPARK_VERSION}
ENV SPARK_HOME=/opt/spark
ENV BIGDL_VERSION=${BIGDL_VERSION}
ENV BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SGX_MEM_SIZE=20GB

ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT

RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 

COPY --from=tini /usr/local/bin/tini /sbin/tini

# Install occlum from package
# RUN apt-get update && \
#     DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#         build-essential ca-certificates openjdk-11-jdk \
#         curl tzdata wget netcat gnupg2 jq make gdb libfuse-dev libtool \
#         libprotobuf-c-dev protobuf-c-compiler libcurl4-openssl-dev libprotobuf-dev unzip
# # Add 01.org & key
# RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu bionic main' | tee /etc/apt/sources.list.d/intelsgx.list && \
#     wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

# RUN echo 'deb [arch=amd64] https://occlum.io/occlum-package-repos/debian bionic main' | tee /etc/apt/sources.list.d/occlum.list && \
#     wget -qO - https://occlum.io/occlum-package-repos/debian/public.key | apt-key add -

# RUN apt-get update && \
#     DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#         build-essential ca-certificates openjdk-11-jdk curl wget netcat net-tools \
#         gnupg2 vim jq make gdb libfuse-dev libtool \
#         libsgx-dcap-ql libsgx-epid libsgx-urts libsgx-quote-ex libsgx-uae-service \
#         libsgx-dcap-quote-verify-dev \
#         occlum && \
#     apt-get clean

# RUN echo "source /etc/profile" >> $HOME/.bashrc 

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-11-jdk && \
    apt-get clean

# prepare Spark
COPY --from=bigdl /opt/spark /opt/spark
COPY --from=bigdl /opt/libhadoop.so /opt/libhadoop.so

# Prepare BigDL
RUN cd /opt && \
    wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
    chmod a+x ./download-bigdl.sh && \
    ./download-bigdl.sh && \
    rm bigdl*.zip

# Prepare Tpc-H
COPY --from=bigdl /opt/zoo-tutorials/tpch-spark/target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar $SPARK_HOME/jars

# Copy scripts & other files
ADD ./run_spark_on_occlum_glibc.sh /opt/run_spark_on_occlum_glibc.sh
ADD ./sqlSuites /opt/sqlSuites

COPY ./entrypoint.sh /opt/

RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/run_spark_on_occlum_glibc.sh

ENTRYPOINT [ "/opt/entrypoint.sh" ]

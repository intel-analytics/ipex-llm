ARG SPARK_VERSION=3.1.3
ARG HADOOP_VERSION=3.2.0

FROM krallin/ubuntu-tini AS tini

FROM ubuntu:20.04 as bigdl

ARG SPARK_VERSION
ARG HADOOP_VERSION
ARG SPARK_SCALA_VERSION=2.12
ENV HADOOP_VERSION=${HADOOP_VERSION}
ENV SPARK_VERSION=${SPARK_VERSION}
ENV SPARK_SCALA_VERSION=${SPARK_SCALA_VERSION}
ENV SPARK_HOME=/opt/spark
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

ARG SPARK_JAR_REPO_URL

ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ENV HTTP_PROXY=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT
ENV HTTPS_PROXY=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk build-essential wget git unzip zip swig libssl-dev cmake && \
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
    rm spark/jars/hive-exec-2.3.7-core.jar && \
    rm spark/jars/slf4j-log4j12-1.7.30.jar && \
    rm spark/jars/log4j-1.2.17.jar && \
    wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar && \
    wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/slf4j/slf4j-reload4j/1.7.35/slf4j-reload4j-1.7.35.jar && \
    wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget -P /opt/spark/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar

# spark modification
RUN cd /opt && \
    wget $SPARK_JAR_REPO_URL/spark-core_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-network-common_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/pyspark.zip && \
    mv /opt/spark-core_2.12-$SPARK_VERSION.jar  /opt/spark/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-kubernetes_2.12-$SPARK_VERSION.jar /opt/spark/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-network-common_2.12-$SPARK_VERSION.jar /opt/spark/jars/spark-network-common_2.12-$SPARK_VERSION.jar && \
    mv /opt/pyspark.zip /opt/spark/python/lib/pyspark.zip

# prepare spark source code and jars for unit test
RUN cd /opt && \
    wget https://github.com/apache/spark/archive/refs/tags/v$SPARK_VERSION.zip && \
    unzip v$SPARK_VERSION.zip && \
    rm v$SPARK_VERSION.zip && \
    mv /opt/spark-$SPARK_VERSION /opt/spark-source && \
    cp -r /opt/spark/bin /opt/spark-source && \
    cd /opt/spark && \
    mkdir /opt/spark/test-jars && \
    cd /opt/spark/test-jars && \
    wget $SPARK_JAR_REPO_URL/spark-core_2.12-$SPARK_VERSION-tests.jar && \
    wget $SPARK_JAR_REPO_URL/spark-catalyst_2.12-$SPARK_VERSION-tests.jar && \
    wget https://repo1.maven.org/maven2/org/scalactic/scalactic_2.12/3.1.4/scalactic_2.12-3.1.4.jar && \
    wget https://repo1.maven.org/maven2/org/scalatest/scalatest_2.12/3.1.4/scalatest_2.12-3.1.4.jar && \
    wget https://repo1.maven.org/maven2/org/mockito/mockito-core/3.4.6/mockito-core-3.4.6.jar && \
    wget https://repo1.maven.org/maven2/com/h2database/h2/1.4.195/h2-1.4.195.jar && \
    wget https://repo1.maven.org/maven2/com/ibm/db2/jcc/11.5.0.0/jcc-11.5.0.0.jar && \
    wget https://repo1.maven.org/maven2/org/apache/parquet/parquet-avro/1.10.1/parquet-avro-1.10.1.jar && \
    wget https://repo1.maven.org/maven2/net/bytebuddy/byte-buddy/1.10.13/byte-buddy-1.10.13.jar && \
    wget https://repo1.maven.org/maven2/org/postgresql/postgresql/42.2.6/postgresql-42.2.6.jar && \
    wget https://repo1.maven.org/maven2/org/scalatestplus/scalatestplus-mockito_2.12/1.0.0-SNAP5/scalatestplus-mockito_2.12-1.0.0-SNAP5.jar && \
    wget https://repo1.maven.org/maven2/org/scalatestplus/scalatestplus-scalacheck_2.12/3.1.0.0-RC2/scalatestplus-scalacheck_2.12-3.1.0.0-RC2.jar && \
    mkdir /opt/spark/test-classes && \
    cd /opt/spark/test-classes && \
    wget $SPARK_JAR_REPO_URL/spark-sql_2.12-$SPARK_VERSION-tests.jar && \
    jar xvf spark-sql_2.12-$SPARK_VERSION-tests.jar && \
    rm spark-sql_2.12-$SPARK_VERSION-tests.jar

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
## 2.4.0 3.1.3 for spark version
RUN cd /opt && \
    git clone https://github.com/intel-analytics/zoo-tutorials.git && \
    cd zoo-tutorials/tpch-spark && \
    sed -i 's/2.11.7/2.12.1/g' tpch.sbt && \
    sed -i 's/2.4.0/3.1.3/g' tpch.sbt && \
    sbt package && \
    cd /opt/zoo-tutorials/tpch-pyspark && \
    zip -r tpch.zip ./tpch

#LGBM make
RUN mkdir /ppml && cd /ppml && \
    git clone --recursive https://github.com/analytics-zoo/LightGBM.git && \
    cd LightGBM && mkdir build && cd build && \
    cmake -DUSE_SWIG=ON -DUSE_SSL=1 .. && make -j4

ADD ./install_python_with_conda.sh /opt
RUN cd /opt && bash ./install_python_with_conda.sh

FROM occlum/occlum:0.29.5-ubuntu20.04 as ppml

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL
ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG SPARK_VERSION
ARG HADOOP_VERSION
ENV HADOOP_VERSION=${HADOOP_VERSION}
ENV SPARK_VERSION=${SPARK_VERSION}
ENV SPARK_HOME=/opt/spark
ENV BIGDL_VERSION=${BIGDL_VERSION}
ENV BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
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
#         build-essential ca-certificates openjdk-8-jdk \
#         curl tzdata wget netcat gnupg2 jq make gdb libfuse-dev libtool \
#         libprotobuf-c-dev protobuf-c-compiler libcurl4-openssl-dev libprotobuf-dev unzip
# # Add 01.org & key
# RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu bionic main' | tee /etc/apt/sources.list.d/intelsgx.list && \
#     wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

# RUN echo 'deb [arch=amd64] https://occlum.io/occlum-package-repos/debian bionic main' | tee /etc/apt/sources.list.d/occlum.list && \
#     wget -qO - https://occlum.io/occlum-package-repos/debian/public.key | apt-key add -

# RUN apt-get update && \
#     DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#         build-essential ca-certificates openjdk-8-jdk curl wget netcat net-tools \
#         gnupg2 vim jq make gdb libfuse-dev libtool \
#         libsgx-dcap-ql libsgx-epid libsgx-urts libsgx-quote-ex libsgx-uae-service \
#         libsgx-dcap-quote-verify-dev \
#         occlum && \
#     apt-get clean

# RUN echo "source /etc/profile" >> $HOME/.bashrc 

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk && \
    apt-get clean

# prepare Spark
COPY --from=bigdl /opt/spark /opt/spark
COPY --from=bigdl /opt/libhadoop.so /opt/libhadoop.so
COPY --from=bigdl /opt/spark-source /opt/spark-source

# Prepare BigDL
RUN cd /opt && \
    wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
    chmod a+x ./download-bigdl.sh && \
    ./download-bigdl.sh && \
    rm bigdl*.zip

# Prepare Tpc-H
COPY --from=bigdl /opt/zoo-tutorials/tpch-spark/target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar $SPARK_HOME/jars

# Prepare pyspark TPC-H
RUN mkdir -p /opt/py-examples/tpch
COPY --from=bigdl /opt/zoo-tutorials/tpch-pyspark/tpch.zip /opt/py-examples/tpch/
COPY --from=bigdl /opt/zoo-tutorials/tpch-pyspark/main.py /opt/py-examples/tpch/

# Copy scripts & other files
ADD ./run_spark_on_occlum_glibc.sh /opt/run_spark_on_occlum_glibc.sh
ADD ./sqlSuites /opt/sqlSuites
ADD ./log4j2.xml /opt/spark/conf/log4j2.xml

# For get_quote C compile
RUN rm /root/demos/remote_attestation/dcap/c_app/dcap_c_test.c
ADD ./dcap_c_test.c /root/demos/remote_attestation/dcap/c_app/
ADD ./dcap-ppml.yaml /root/demos/remote_attestation/dcap/
ADD ./get_quote_on_ppml.sh /root/demos/remote_attestation/dcap

# For attestation
ADD ./verify-attestation-service.sh /opt
ADD ./print_enclave_signer.sh /opt
ADD ./register.sh /opt

COPY ./entrypoint.sh /opt/
COPY ./ehsm_entry.sh /opt/
COPY ./spark.yaml /opt/

RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/ehsm_entry.sh && \
    chmod a+x /opt/run_spark_on_occlum_glibc.sh && \
    chmod a+x /opt/verify-attestation-service.sh && \
    chmod a+x /opt/print_enclave_signer.sh && \
    chmod a+x /opt/register.sh && \
    chmod a+x /root/demos/remote_attestation/dcap/get_quote_on_ppml.sh && \
    chmod a+x /root/demos/remote_attestation/dcap/c_app/dcap_c_test.c && \
    sed -i '/aesm/d' /root/.bashrc && \
    rm -rf /var/run/aesmd

# Add python lib
ADD ./python-glibc.yaml /opt/python-glibc.yaml
COPY --from=bigdl /opt/python-occlum /opt/python-occlum
ADD ./py-examples /opt/py-examples

# replace lightgbmlib.jar
RUN rm -rf $BIGDL_HOME/jars/lightgbmlib*
COPY --from=bigdl /ppml/LightGBM/build/lightgbmlib.jar $BIGDL_HOME/jars/
ENTRYPOINT [ "/opt/entrypoint.sh" ]

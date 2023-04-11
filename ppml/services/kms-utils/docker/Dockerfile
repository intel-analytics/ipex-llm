ARG JDK_VERSION=8u192
ARG SPARK_VERSION=3.1.3
ARG BIGDL_VERSION=2.3.0-SNAPSHOT

# stage.1 java & spark
FROM ubuntu:20.04 AS builder

ARG SPARK_VERSION
ARG JDK_VERSION
ARG JDK_URL
ENV JDK_VERSION         ${JDK_VERSION}
ENV SPARK_VERSION       ${SPARK_VERSION}
ENV JAVA_HOME           /opt/jdk${JDK_VERSION}
ENV SPARK_HOME          /opt/spark-${SPARK_VERSION}
ENV PATH                ${JAVA_HOME}/bin:${SPARK_HOME}/bin:${PATH}

RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt-get install -y apt-utils wget unzip patch zip

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
    # remove log4j 1.x jars
    rm -f ${SPARK_HOME}/jars/log4j-1.2.17.jar && \
    rm -f ${SPARK_HOME}/jars/slf4j-log4j12-1.7.16.jar && \
    rm -f ${SPARK_HOME}/jars/apache-log4j-extras-1.2.17.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/slf4j/slf4j-reload4j/1.7.35/slf4j-reload4j-1.7.35.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget -P ${SPARK_HOME}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar
    
ADD https://raw.githubusercontent.com/intel-analytics/BigDL/main/docker/bigdl-k8s/log4j2.xml ${SPARK_HOME}/conf/log4j2.xml


# stage.2 bigdl
FROM ubuntu:20.04 as bigdl
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /opt/bigdl-${BIGDL_VERSION}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip git
RUN wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
    chmod a+x ./download-bigdl.sh
RUN ./download-bigdl.sh && \
    rm bigdl*.zip


# stage.3 kms-util
FROM ubuntu:20.04
ARG EHSM_KMS_BRANCH_VERSION=main
ARG SPARK_VERSION
ARG BIGDL_VERSION
ENV BIGDL_VERSION       ${BIGDL_VERSION}
ENV SPARK_VERSION       ${SPARK_VERSION}
ENV JAVA_HOME           /opt/jdk8
ENV SPARK_HOME          /opt/spark-${SPARK_VERSION}
ENV BIGDL_HOME          /opt/bigdl-${BIGDL_VERSION}
ENV PATH                ${JAVA_HOME}/bin:${SPARK_HOME}/bin:${PATH}

# Step 1. Install requirement tools
RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt-get install -y vim autoconf automake build-essential cmake curl debhelper git libcurl4-openssl-dev libprotobuf-dev libssl-dev libtool lsb-release ocaml ocamlbuild protobuf-compiler python wget libcurl4 libprotobuf17 libssl1.1 make kmod g++ libjsoncpp-dev uuid-dev openjdk-8-jdk unzip && \
# Step 2. Install Intel SGX SDK
    mkdir /opt/intel && cd /opt/intel && \
    wget https://download.01.org/intel-sgx/sgx-dcap/1.13/linux/distro/ubuntu20.04-server/sgx_linux_x64_sdk_2.16.100.4.bin && \
    chmod a+x ./sgx_linux_x64_sdk_2.16.100.4.bin && \
    printf "no\n/opt/intel\n"|./sgx_linux_x64_sdk_2.16.100.4.bin && \
    . /opt/intel/sgxsdk/environment && \
# Step 3. Install DCAP packages and set PCCS
# DCAP repository setup
    wget https://download.01.org/intel-sgx/sgx-dcap/1.12.1/linux/distro/ubuntu20.04-server/sgx_debian_local_repo.tgz && \
    tar xzf sgx_debian_local_repo.tgz && \
    echo 'deb [trusted=yes arch=amd64] file:///opt/intel/sgx_debian_local_repo focal main' | tee /etc/apt/sources.list.d/intel-sgx.list && \
    wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add - && \
    apt-get update --fix-missing && \
    apt-get install -y libsgx-enclave-common-dev libsgx-ae-qe3 libsgx-ae-qve libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-dcap-quote-verify-dev libsgx-dcap-default-qpl-dev libsgx-quote-ex-dev libsgx-uae-service libsgx-ra-network libsgx-ra-uefi libsgx-dcap-ql-dev

ADD ./entrypoint.sh /home/entrypoint.sh
COPY --from=builder /opt/jdk /opt/jdk8
COPY --from=builder /opt/spark-${SPARK_VERSION} /opt/spark-${SPARK_VERSION}
COPY --from=bigdl   ${BIGDL_HOME} ${BIGDL_HOME}

# python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install cryptography && \
    pip3 install requests && \
    pip install setuptools==58.4.0 

CMD ["sh", "sleep 10s"]

ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2

# stage.1 graphene
FROM ubuntu:18.04 AS graphene
ARG GRAPHENE_BRANCH=branch-1.2-rc1
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        autoconf bison build-essential coreutils gawk git wget \
        python3 python3-pip libcurl4-openssl-dev \
        libprotobuf-c-dev protobuf-c-compiler python3-protobuf wget
RUN git clone https://github.com/analytics-zoo/graphene.git /graphene
RUN cd /graphene && \
    git fetch origin $GRAPHENE_BRANCH && \
    git checkout $GRAPHENE_BRANCH
RUN pip3 install ninja meson && \
    python3 -m pip install toml==0.10.2 click jinja2
RUN cd /graphene/Pal/src/host/Linux-SGX && \
    git clone https://github.com/intel/SGXDataCenterAttestationPrimitives.git linux-sgx-driver && \
    cd linux-sgx-driver && \
    git checkout DCAP_1.7 && \
    cp -r driver/linux/* .
RUN cd /graphene && \
    make && \
    ISGX_DRIVER_PATH=/graphene/Pal/src/host/Linux-SGX/linux-sgx-driver make -s -j4 SGX=1 && \
    meson setup build/ --prefix="/graphene/meson_build_output" \
    --buildtype=release -Ddirect=enabled -Dsgx=enabled  && \
    ninja -C build/ && \
    ninja -C build/ install
RUN for f in $(find /graphene/Runtime -type l); do cp --remove-destination $(realpath $f) $f; done

ENV PATH="/graphene/build/Runtime:${PATH}" 
ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2
ARG HADOOP_VERSION=3.2.0
ENV BIGDL_VERSION=${BIGDL_VERSION}
ENV BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SGX_MEM_SIZE=20GB

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils unzip

# JDK
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-11-jdk && \
    apt-get clean

# BigDL
RUN cd /opt && \
    wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
    chmod a+x ./download-bigdl.sh && \
    ./download-bigdl.sh && \
    rm bigdl*.zip

WORKDIR /opt
ADD bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar /opt/bigdl-0.14.0-SNAPSHOT/jars
ADD runFlserver.sh /opt
ADD runHflClient1.sh /opt
ADD runHflClient2.sh /opt
ADD runVflClient1.sh /opt
ADD runVflClient2.sh /opt
ADD ./data /opt/data
ADD ppml-conf.yaml /opt
ADD java.manifest.sgx /opt
ADD Makefile /opt

RUN chmod a+x runFlserver.sh && \
    chmod a+x runHflClient1.sh && \
    chmod a+x runHflClient2.sh && \
    chmod a+x runVflClient1.sh && \
    chmod a+x runVflClient2.sh

ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2

# stage.1 graphene
FROM ubuntu:18.04 AS graphene
ARG GRAPHENE_BRANCH=branch-1.2-rc1
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        autoconf bison build-essential coreutils gawk git wget \
        python3 python3-pip libcurl4-openssl-dev \
        libprotobuf-c-dev protobuf-c-compiler python3-protobuf wget build-essential
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
RUN /graphene/Scripts/download --output redis.tar.gz --sha256 f7ded6c27d48c20bc78e797046c79b6bc411121f0c2d7eead9fea50d6b0b6290 --url https://github.com/antirez/redis/archive/6.0.5.tar.gz
RUN for f in $(find /graphene/Runtime -type l); do cp --remove-destination $(realpath $f) $f; done

# stage.2 jdk &
FROM ubuntu:18.04 as spark
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils wget unzip
# java
RUN wget $JDK_URL -O jdk-$JDK_VERSION-linux-x64.tar.gz && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk8

ENV JAVA_HOME                           /opt/jdk8
ENV PATH                                /graphene/meson_build_output/bin:${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP                            127.0.0.1
ENV SGX_MEM_SIZE                        64G
ARG SPARK_VERSION=3.1.2
RUN mkdir -p /graphene && \
    mkdir -p /graphene/Runtime && \
    mkdir -p /graphene/python && \
    mkdir -p /graphene/Tools && \
    mkdir -p /graphene/Pal/src && \
    mkdir -p /graphene/meson_build_output

COPY --from=graphene /graphene/Scripts /graphene/Scripts
COPY --from=graphene /graphene/Runtime/ /graphene/Runtime
COPY --from=graphene /graphene/python /graphene/python
COPY --from=graphene /graphene/Pal /graphene/Pal
COPY --from=graphene /graphene/Pal/src/host/Linux-SGX/generated_offsets.py /graphene/python/
COPY --from=graphene /graphene/Tools/argv_serializer /graphene/Tools
COPY --from=graphene /graphene/meson_build_output /graphene/meson_build_output

RUN mkdir /opt/bigdl-0.14.0-SNAPSHOT && mkdir /opt/bigdl-0.14.0-SNAPSHOT/jars

WORKDIR /opt
ADD bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar /opt/bigdl-0.14.0-SNAPSHOT/jars
ADD runFlServer.sh /opt
ADD runHflClient1.sh /opt
ADD runHflClient2.sh /opt
ADD runVflClient1.sh /opt
ADD runVflClient2.sh /opt
ADD data /opt/data
ADD ppml-conf.yaml /opt
ADD Makefile /opt
ADD bash.manifest.template /opt

# spark
ARG SPARK_JAR_REPO_URL
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

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git tree && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y openssl libssl-dev pkg-config && \
    apt-get install -y netcat net-tools && \
    apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir ninja meson && \
    python3 -m pip install toml==0.10.2 click jinja2
ENV PYTHONPATH  ${PYTHONPATH}:/graphene/meson_build_output/lib/python3.6/site-packages
ENV LC_ALL      C.UTF-8
ENV LANG        C.UTF-8
#generate enclave-key
RUN openssl genrsa -3 -out /graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem 3072

# build bash.manifest.sgx
RUN cd /opt && make SGX=1 DEBUG=1

RUN chmod a+x runFlServer.sh && \
    chmod a+x runHflClient1.sh && \
    chmod a+x runHflClient2.sh && \
    chmod a+x runVflClient1.sh && \
    chmod a+x runVflClient2.sh
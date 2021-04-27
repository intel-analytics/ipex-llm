# stage.1 graphene
FROM ubuntu:18.04 AS graphene

RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y autoconf bison build-essential coreutils gawk git libcurl4-openssl-dev libprotobuf-c-dev protobuf-c-compiler python3-protobuf wget
RUN git clone https://github.com/analytics-zoo/graphene.git /graphene
RUN cd /graphene && \
    git fetch origin branch-0.2 && \
    git checkout branch-0.2
RUN cd /graphene/Pal/src/host/Linux-SGX && \
    git clone https://github.com/intel/SGXDataCenterAttestationPrimitives.git linux-sgx-driver && \
    cd linux-sgx-driver && \
    git checkout DCAP_1.7 && \
    cp -r driver/linux/* .
RUN cd /graphene && \
    ISGX_DRIVER_PATH=/graphene/Pal/src/host/Linux-SGX/linux-sgx-driver make -s -j4 SGX=1 WERROR=1 && true
RUN for f in $(find /graphene/Runtime -type l); do cp --remove-destination $(realpath $f) $f; done
RUN /graphene/Scripts/download --output redis.tar.gz --sha256 f7ded6c27d48c20bc78e797046c79b6bc411121f0c2d7eead9fea50d6b0b6290 --url https://github.com/antirez/redis/archive/6.0.5.tar.gz

# stage.2 jdk & flink
FROM ubuntu:18.04 as flink
ARG FLINK_VERSION=1.10.1
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ENV FLINK_VERSION ${FLINK_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils wget unzip
# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk
# flink
RUN cd /opt && \
    wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar -zxvf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz
RUN ls -al /opt

# stage.3 analytics-zoo
FROM ubuntu:18.04 as analytics-zoo
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG ANALYTICS_ZOO_VERSION=0.11.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.2
ARG SPARK_VERSION=2.4.3
ARG SPARK_MAJOR_VERSION=2.4
ENV ANALYTICS_ZOO_VERSION	${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV JAVA_HOME			/opt/jdk8
ENV PATH			${JAVA_HOME}/bin:${PATH}

COPY --from=flink /opt/jdk  /opt/jdk8

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip git maven
RUN git clone https://github.com/intel-analytics/analytics-zoo.git && \
    cd analytics-zoo/zoo && \
    export MAVEN_OPTS="-XX:ReservedCodeCacheSize=512m -XX:MaxPermSize=3G \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    mvn clean package -DskipTests -Dspark.version=${SPARK_VERSION} \
        -Dbigdl.artifactId=bigdl-SPARK_$SPARK_MAJOR_VERSION -P spark_2.4+ -Dbigdl.version=${BIGDL_VERSION}
RUN cd /opt && \
    mkdir resnet50 && \
    cd resnet50 && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.bin/download" -O resnet_v1_50.bin && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.xml/download" -O resnet_v1_50.xml

# stage.4 az ppml
FROM ubuntu:18.04
ARG ANALYTICS_ZOO_VERSION=0.11.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.2
ARG SPARK_VERSION=2.4.3
ARG FLINK_VERSION=1.10.1
ENV ANALYTICS_ZOO_VERSION		${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION			${SPARK_VERSION}
ENV BIGDL_VERSION			${BIGDL_VERSION}
ENV JAVA_HOME				/opt/jdk8
ENV PATH				${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP				127.0.0.1
ENV SGX_MEM_SIZE			64G
ENV REDIS_PORT				6379
ENV FLINK_VERSION			${FLINK_VERSION}
ENV FLINK_HOME				/ppml/trusted-realtime-ml/java/work/flink-${FLINK_VERSION}
ENV FLINK_JOB_MANAGER_IP		127.0.0.1
ENV FLINK_JOB_MANAGER_REST_PORT		8081
ENV FLINK_JOB_MANAGER_RPC_PORT		6123
ENV FLINK_TASK_MANAGER_IP		127.0.0.1
ENV FLINK_TASK_MANAGER_DATA_PORT	6124
ENV FLINK_TASK_MANAGER_RPC_PORT		6125
ENV FLINK_TASK_MANAGER_TASKSLOTS_NUM	1
ENV CORE_NUM                            2

RUN mkdir -p /graphene && \
    mkdir -p /graphene/Runtime && \
    mkdir -p /graphene/python && \
    mkdir -p /graphene/Tools && \
    mkdir -p /graphene/Pal/src && \
    mkdir -p /ppml/trusted-realtime-ml/redis && \
    mkdir -p /ppml/trusted-realtime-ml/redis/work/keys && \
    mkdir -p /ppml/trusted-realtime-ml/redis/work/password && \
    mkdir -p /ppml/trusted-realtime-ml/java/work && \
    mkdir -p /ppml/trusted-realtime-ml/java/work/keys && \
    mkdir -p /ppml/trusted-realtime-ml/java/work/password && \
    mkdir -p /ppml/trusted-realtime-ml/java/work/data && \
    mkdir -p /ppml/trusted-realtime-ml/java/work/models && \
    mkdir -p /graphene/Pal/src/host/Linux-SGX/signer

COPY --from=graphene /graphene/Scripts /graphene/Scripts
COPY --from=graphene /graphene/Runtime/ /graphene/Runtime
COPY --from=graphene /graphene/python /graphene/python
COPY --from=graphene /graphene/Pal /graphene/Pal
COPY --from=graphene /graphene/Pal/src/host/Linux-SGX/generated_offsets.py /graphene/python/
COPY --from=graphene /graphene/Examples/common_tools/get_deps.sh /graphene/get_deps.sh
COPY --from=graphene /graphene/Tools/argv_serializer /graphene/Tools
COPY --from=graphene /redis.tar.gz /ppml/trusted-realtime-ml/redis
COPY --from=flink /opt/jdk  /opt/jdk8
COPY --from=flink /opt/flink-${FLINK_VERSION} /ppml/trusted-realtime-ml/java/work/flink-${FLINK_VERSION}
COPY --from=analytics-zoo /analytics-zoo/zoo/target/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar /ppml/trusted-realtime-ml/java/work/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar
COPY --from=analytics-zoo /analytics-zoo/zoo/target/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar /ppml/trusted-realtime-ml/java/work/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar
COPY --from=analytics-zoo /opt/resnet50/ /ppml/trusted-realtime-ml/java/work/models/resnet50

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git tree && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y openssl libssl-dev pkg-config && \
    apt-get install -y netcat net-tools

RUN mkdir -p /ppml/trusted-realtime-ml/redis/src && \
    tar -C /ppml/trusted-realtime-ml/redis/src --strip-components=1 -xf /ppml/trusted-realtime-ml/redis/redis.tar.gz && \
    make BUILD_TLS=yes -C /ppml/trusted-realtime-ml/redis/src && \
    cp /ppml/trusted-realtime-ml/redis/src/src/redis-server /ppml/trusted-realtime-ml/redis/redis-server

ADD ./redis.Makefile /ppml/trusted-realtime-ml/redis/Makefile
ADD ./redis-server.manifest.template /ppml/trusted-realtime-ml/redis/redis-server.manifest.template
ADD ./init-redis.sh /ppml/trusted-realtime-ml/redis/init-redis.sh
ADD ./start-redis.sh /ppml/trusted-realtime-ml/redis/start-redis.sh
ADD ./java.Makefile /ppml/trusted-realtime-ml/java/Makefile
ADD ./java.manifest.template /ppml/trusted-realtime-ml/java/java.manifest.template
ADD ./init-java.sh /ppml/trusted-realtime-ml/java/init-java.sh
ADD ./start-flink-jobmanager.sh /ppml/trusted-realtime-ml/java/start-flink-jobmanager.sh
ADD ./start-flink-taskmanager.sh /ppml/trusted-realtime-ml/java/start-flink-taskmanager.sh
ADD ./start-http-frontend.sh /ppml/trusted-realtime-ml/java/start-http-frontend.sh
ADD ./cluster-serving-config.yaml /ppml/trusted-realtime-ml/java/config.yaml
ADD ./init-cluster-serving.sh /ppml/trusted-realtime-ml/java/init-cluster-serving.sh
ADD ./start-cluster-serving-job.sh /ppml/trusted-realtime-ml/java/start-cluster-serving-job.sh
ADD ./check-status.sh /ppml/trusted-realtime-ml/check-status.sh
ADD ./start-all.sh /ppml/trusted-realtime-ml/start-all.sh

RUN chmod a+x /ppml/trusted-realtime-ml/redis/init-redis.sh && \
    chmod a+x /ppml/trusted-realtime-ml/redis/start-redis.sh && \
    chmod a+x /ppml/trusted-realtime-ml/java/init-java.sh && \
    chmod a+x /ppml/trusted-realtime-ml/java/start-flink-jobmanager.sh && \
    chmod a+x /ppml/trusted-realtime-ml/java/start-flink-taskmanager.sh && \
    chmod a+x /ppml/trusted-realtime-ml/java/start-http-frontend.sh && \
    chmod a+x /ppml/trusted-realtime-ml/java/init-cluster-serving.sh && \
    chmod a+x /ppml/trusted-realtime-ml/java/start-cluster-serving-job.sh && \
    chmod a+x /ppml/trusted-realtime-ml/check-status.sh && \
    chmod a+x /ppml/trusted-realtime-ml/start-all.sh

# stage.1 Redis
FROM ubuntu:18.04 as redis-tls

WORKDIR /opt

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        git build-essential coreutils ca-certificates && \
    apt-get clean

RUN git clone https://github.com/openssl/openssl.git && \
    cd openssl && \
    git checkout tags/OpenSSL_1_1_1 -b OpenSSL_1_1_1 && \
    ./config \
    --openssldir=/opt/ssl \
    --with-rand-seed=rdcpu \
    no-zlib no-async no-tests && \
    make -j `getconf _NPROCESSORS_ONLN` && make install

RUN git clone https://github.com/redis/redis.git && \
    cd redis && \
    git checkout 6.0.6 && \
    make -j `getconf _NPROCESSORS_ONLN` BUILD_TLS=yes && make PREFIX=/opt/redis install

# stage. 2 Flink & analytics-zoo
FROM ubuntu:18.04 as analytics-zoo

ARG ANALYTICS_ZOO_VERSION=0.11.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.2
ARG SPARK_VERSION=2.4.3
ARG SPARK_MAJOR_VERSION=2.4
ARG FLINK_VERSION=1.10.1
ENV ANALYTICS_ZOO_VERSION		${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME				/usr/lib/jvm/java-8-openjdk-amd64
ENV FLINK_VERSION			${FLINK_VERSION}
ENV FLINK_HOME				/opt/flink-${FLINK_VERSION}
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        git build-essential coreutils ca-certificates openjdk-8-jdk wget unzip apt-utils curl maven && \
    apt-get clean

# Download flink
RUN cd /opt && \
    wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar -zxvf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz

# analytics-zoo
RUN cd /opt && git clone https://github.com/intel-analytics/analytics-zoo.git && \
    cd analytics-zoo/zoo && \
    export MAVEN_OPTS="-XX:ReservedCodeCacheSize=512m -XX:MaxPermSize=3G \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    mvn clean package -DskipTests -Dspark.version=${SPARK_VERSION} \
        -Dbigdl.artifactId=bigdl-SPARK_$SPARK_MAJOR_VERSION -P spark_2.4+ -Dbigdl.version=${BIGDL_VERSION}

# models
RUN cd /opt && \
    mkdir resnet50 && \
    cd resnet50 && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.bin/download" -O resnet_v1_50.bin && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.xml/download" -O resnet_v1_50.xml

# stage.3 az ppml occlum
FROM occlum/occlum:0.21.0-ubuntu18.04

ARG ANALYTICS_ZOO_VERSION=0.11.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.2
ARG SPARK_VERSION=2.4.3
ARG FLINK_VERSION=1.10.1
ENV ANALYTICS_ZOO_VERSION		${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION			${SPARK_VERSION}
ENV BIGDL_VERSION			${BIGDL_VERSION}
ENV JAVA_HOME				/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH				${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP				127.0.0.1
ENV SGX_MEM_SIZE			64G
ENV REDIS_PORT				6379
ENV FLINK_VERSION			${FLINK_VERSION}
ENV FLINK_HOME				/opt/flink-${FLINK_VERSION}
ENV FLINK_JOB_MANAGER_IP		127.0.0.1
ENV FLINK_JOB_MANAGER_REST_PORT		8081
ENV FLINK_JOB_MANAGER_RPC_PORT		6123
ENV FLINK_TASK_MANAGER_IP		127.0.0.1
ENV FLINK_TASK_MANAGER_DATA_PORT	6124
ENV FLINK_TASK_MANAGER_RPC_PORT		6125
ENV FLINK_TASK_MANAGER_TASKSLOTS_NUM	1
ENV CORE_NUM                            2

RUN mkdir -p /opt/analytics-zoo

COPY --from=redis-tls /opt/ssl /opt/ssl
COPY --from=redis-tls /opt/redis /opt/redis

COPY --from=analytics-zoo /opt/flink-${FLINK_VERSION} /opt/flink-${FLINK_VERSION}
COPY --from=analytics-zoo /opt/resnet50 /opt/resnet50
COPY --from=analytics-zoo /opt/analytics-zoo/zoo/target/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar /opt/analytics-zoo/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar
COPY --from=analytics-zoo /opt/analytics-zoo/zoo/target/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar /opt/analytics-zoo/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar

# Add key for SGX repo
RUN wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        build-essential ca-certificates openjdk-11-jdk curl wget netcat net-tools && \
    apt-get clean

# Cluster Serving config
ADD ./cluster-serving-config.yaml /opt/config.yaml
# PPML scripts
ADD ./init-occlum-taskmanager.sh /opt/init-occlum-taskmanager.sh
ADD ./start-redis.sh /opt/start-redis.sh
ADD ./check-status.sh /opt/check-status.sh
ADD ./start-flink-jobmanager.sh /opt/start-flink-jobmanager.sh
ADD ./start-flink-taskmanager.sh /opt/start-flink-taskmanager.sh
ADD ./hosts /opt/hosts
ADD ./start-cluster-serving-job.sh /opt/start-cluster-serving-job.sh
ADD ./start-http-frontend.sh /opt/start-http-frontend.sh
ADD ./start-all.sh /opt/start-all.sh
# PPML Start & Stop shell
ADD ./start-local-cluster-serving.sh /opt/start-local-cluster-serving.sh
ADD ./start-distributed-cluster-serving.sh /opt/start-distributed-cluster-serving.sh
ADD ./stop-distributed-cluster-serving.sh /opt/stop-distributed-cluster-serving.sh

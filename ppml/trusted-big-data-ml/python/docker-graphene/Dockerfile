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

# stage.2 jdk & spark
FROM ubuntu:18.04 as spark
ARG SPARK_VERSION=2.4.3
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ENV SPARK_VERSION	${SPARK_VERSION}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils wget unzip patch zip
ADD rdd.patch /opt/rdd.patch
ADD shuffle.patch /opt/shuffle.patch
# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk
# spark
RUN cd /opt && \
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    wget -O spark-network-common_2.11-${SPARK_VERSION}.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.11-${SPARK_VERSION}.jar && \
    mv spark-network-common_2.11-${SPARK_VERSION}.jar spark-${SPARK_VERSION}/jars && \
    mv spark-${SPARK_VERSION}/python/lib/pyspark.zip spark-${SPARK_VERSION}/python/lib/pyspark.zip.bac && \
    patch spark-${SPARK_VERSION}/python/pyspark/rdd.py /opt/rdd.patch && \
    patch spark-${SPARK_VERSION}/python/pyspark/shuffle.py /opt/shuffle.patch && \
    cd spark-${SPARK_VERSION}/python && \
    zip -r lib/pyspark.zip pyspark

# stage.3 bigdl
FROM ubuntu:18.04 AS bigdl
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ENV JAVA_HOME           /opt/jdk8
ENV PATH                ${JAVA_HOME}/bin:${PATH}

ADD bigdl.lenet.training.patch /bigdl.lenet.training.patch
COPY --from=spark /opt/jdk  /opt/jdk8

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils maven git

#bigdl
RUN git clone https://github.com/intel-analytics/BigDL.git && \
    cd BigDL && \
    git apply /bigdl.lenet.training.patch && \
    git status && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    bash make-dist.sh -P spark_2.x

# stage.4 analytics-zoo
FROM ubuntu:18.04 as analytics-zoo
ARG ANALYTICS_ZOO_VERSION=0.11.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.2
ARG SPARK_VERSION=2.4.3
ENV ANALYTICS_ZOO_VERSION	${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV ANALYTICS_ZOO_HOME		/analytics-zoo-${ANALYTICS_ZOO_VERSION}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip
RUN wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/master/docker/zoo/download-analytics-zoo.sh && \
    chmod a+x ./download-analytics-zoo.sh
RUN ./download-analytics-zoo.sh

# stage.5 az ppml
FROM ubuntu:18.04
ARG ANALYTICS_ZOO_VERSION=0.11.0-SNAPSHOT
ARG SPARK_VERSION=2.4.3
ENV ANALYTICS_ZOO_VERSION		${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION			${SPARK_VERSION}
ENV SPARK_HOME				/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME			/ppml/trusted-big-data-ml/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME				/opt/jdk8
ENV PATH				${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP				127.0.0.1
ENV SGX_MEM_SIZE			64G
ENV SPARK_MASTER_IP			127.0.0.1
ENV SPARK_MASTER_PORT			7077
ENV SPARK_MASTER_WEBUI_PORT		8080
ENV SPARK_MASTER			spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT
ENV SPARK_WORKER_PORT			8082
ENV SPARK_WORKER_WEBUI_PORT		8081
ENV SPARK_DRIVER_PORT			10027
ENV SPARK_DRIVER_BLOCK_MANAGER_PORT	10026
ENV SPARK_DRIVER_IP			$LOCAL_IP
ENV SPARK_BLOCK_MANAGER_PORT		10025

RUN mkdir -p /graphene && \
    mkdir -p /graphene/Runtime && \
    mkdir -p /graphene/python && \
    mkdir -p /graphene/Tools && \
    mkdir -p /graphene/Pal/src && \
    mkdir -p /ppml/trusted-big-data-ml/work && \
    mkdir -p /ppml/trusted-big-data-ml/work/keys && \
    mkdir -p /ppml/trusted-big-data-ml/work/password && \
    mkdir -p /ppml/trusted-big-data-ml/work/data && \
    mkdir -p /ppml/trusted-big-data-ml/work/models && \
    mkdir -p /ppml/trusted-big-data-ml/work/apps && \
    mkdir -p /graphene/Pal/src/host/Linux-SGX/signer

COPY --from=graphene /graphene/Scripts /graphene/Scripts
COPY --from=graphene /graphene/Runtime/ /graphene/Runtime
COPY --from=graphene /graphene/python /graphene/python
COPY --from=graphene /graphene/Pal /graphene/Pal
COPY --from=graphene /graphene/Pal/src/host/Linux-SGX/generated_offsets.py /graphene/python/
COPY --from=graphene /graphene/Examples/common_tools/get_deps.sh /graphene/get_deps.sh
COPY --from=graphene /graphene/Tools/argv_serializer /graphene/Tools
COPY --from=spark /opt/jdk  /opt/jdk8
COPY --from=spark /opt/spark-${SPARK_VERSION} /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}
COPY --from=bigdl /BigDL/dist/lib/bigdl-*-jar-with-dependencies.jar /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar
COPY --from=analytics-zoo /analytics-zoo-${ANALYTICS_ZOO_VERSION} /ppml/trusted-big-data-ml/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git tree && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y netcat net-tools

#python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install numpy scipy && \
    pip install --no-binary pandas -I pandas && \
    pip install scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0 && \
    pip install torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install tornado && \
    python3 -m ipykernel.kernelspec

ADD ./bash.manifest.template /ppml/trusted-big-data-ml/bash.manifest.template
ADD ./Makefile /ppml/trusted-big-data-ml/Makefile
ADD ./init.sh /ppml/trusted-big-data-ml/init.sh
ADD ./clean.sh /ppml/trusted-big-data-ml/clean.sh
ADD ./examples /ppml/trusted-big-data-ml/work/examples
ADD ./start-spark-local-train-sgx.sh /ppml/trusted-big-data-ml/start-spark-local-train-sgx.sh
ADD ./start-spark-standalone-master-sgx.sh /ppml/trusted-big-data-ml/start-spark-standalone-master-sgx.sh
ADD ./start-spark-standalone-worker-sgx.sh /ppml/trusted-big-data-ml/start-spark-standalone-worker-sgx.sh
ADD ./start-spark-standalone-driver-sgx.sh /ppml/trusted-big-data-ml/start-spark-standalone-driver-sgx.sh
ADD ./ppml-spark-submit.sh /ppml/trusted-big-data-ml/ppml-spark-submit.sh
ADD ./check-status.sh /ppml/trusted-big-data-ml/check-status.sh

RUN chmod a+x /ppml/trusted-big-data-ml/init.sh && \
    chmod a+x /ppml/trusted-big-data-ml/clean.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-local-train-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-standalone-master-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-standalone-worker-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-standalone-driver-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/ppml-spark-submit.sh && \
    chmod a+x /ppml/trusted-big-data-ml/check-status.sh

    


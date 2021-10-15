FROM occlum/occlum:0.23.7-ubuntu18.04 AS occlum

ENV LOCAL_IP your_local_ip
ARG BIGDL_VERSION=0.13.0
ARG SPARK_VERSION=2.4.6
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV http_proxy $http_proxy
ENV https_proxy $https_proxy

RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk unzip

# Prepare Spark
RUN mkdir -p /bin/examples/jars
RUN mkdir -p /ppml/docker-occlum && cd /ppml/docker-occlum && \
    wget -P /ppml/docker-occlum https://archive.apache.org/dist/spark/spark-2.4.6/spark-2.4.6-bin-hadoop2.7.tgz && \
    tar -xvzf /ppml/docker-occlum/spark-2.4.6-bin-hadoop2.7.tgz  &&  \
    wget -P /bin/examples/jars/spark-network-common_2.11-2.4.6.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.11-2.4.6.jar && \
    rm /ppml/docker-occlum/spark-2.4.6-bin-hadoop2.7.tgz

ADD ./hosts /ppml/docker-occlum/hosts
ADD ./README.md  /ppml/docker-occlum/README.md 
ADD ./run_spark_on_occlum_glibc.sh /ppml/docker-occlum/run_spark_on_occlum_glibc.sh
ADD ./start-spark-local.sh /ppml/docker-occlum/start-spark-local.sh

RUN wget -P /ppml/docker-occlum https://repo1.maven.org/maven2/com/intel/analytics/bigdl/dist-spark-${SPARK_VERSION}-scala-2.11.8-all/${BIGDL_VERSION}/dist-spark-${SPARK_VERSION}-scala-2.11.8-all-${BIGDL_VERSION}-dist.zip  && \
    unzip /ppml/docker-occlum/dist-spark-${SPARK_VERSION}-scala-2.11.8-all-${BIGDL_VERSION}-dist.zip -d /ppml/docker-occlum/bigdl && \
    rm /ppml/docker-occlum/dist-spark-${SPARK_VERSION}-scala-2.11.8-all-${BIGDL_VERSION}-dist.zip && \
    cp /ppml/docker-occlum/bigdl/lib/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar /ppml/docker-occlum/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar && \
    rm -r /ppml/docker-occlum/bigdl

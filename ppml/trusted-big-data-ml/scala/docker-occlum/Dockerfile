FROM occlum/occlum:0.24.0-ubuntu18.04 AS occlum

ENV LOCAL_IP your_local_ip
ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV http_proxy $http_proxy
ENV https_proxy $https_proxy

RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk unzip

# Prepare Spark
RUN mkdir -p /bin/examples/jars

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz -P /opt/ && \
    tar -zxvf /opt/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop2.7 /opt/spark && \
    rm /opt/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    cp -r /opt/spark/examples /bin/examples && \
    cp -r /opt/spark/kubernetes/tests /opt/spark/tests

# Remove fork with libhadoop.so and spark-network-common.jar
RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so -P /lib/ && \
    rm -f /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar && \
    wget -O /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.12-${SPARK_VERSION}.jar

ADD ./run_spark_on_occlum_glibc.sh /opt/run_spark_on_occlum_glibc.sh

RUN wget -P /opt/ https://repo1.maven.org/maven2/com/intel/analytics/bigdl/dist-spark-${SPARK_VERSION}-scala-2.11.8-all/${BIGDL_VERSION}/dist-spark-${SPARK_VERSION}-scala-2.11.8-all-${BIGDL_VERSION}-dist.zip  && \
    unzip /opt/dist-spark-${SPARK_VERSION}-scala-2.11.8-all-${BIGDL_VERSION}-dist.zip -d /opt/bigdl && \
    rm /opt/dist-spark-${SPARK_VERSION}-scala-2.11.8-all-${BIGDL_VERSION}-dist.zip && \
    cp /opt/bigdl/lib/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar /opt/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar && \
    rm -r /opt/bigdl

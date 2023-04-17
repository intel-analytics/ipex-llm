ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG SPARK_VERSION=3.1.3
ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.3.0-SNAPSHOT

# java
FROM ubuntu:20.04 as java
ARG JDK_VERSION=8u192
ARG JDK_URL
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /bigdl-${BIGDL_VERSION}
ARG http_proxy
ARG https_proxy

RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-utils wget unzip && \
    wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

# gramine & sqlite
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
USER root
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /ppml/bigdl-${BIGDL_VERSION}
ARG http_proxy
ARG https_proxy
ADD ./bigdl-kms-entrypoint.sh /ppml/bigdl-kms-entrypoint.sh
RUN mkdir -p /ppml/keys && \
    mkdir -p /ppml/password && \
    mkdir -p /ppml/data && \
    cd /ppml && \
    chmod a+x /ppml/start-service.sh && \
    gramine-argv-serializer bash -c '$sgx_command' > /ppml/secured_argvs && \
    apt-get install sqlite3
COPY --from=java /opt/jdk  /opt/jdk8
# ppml
COPY --from=intelanalytics/bigdl-ppml-trusted-bigdata-gramine-reference-8g:2.3.0-SNAPSHOT ${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar /ppml/jars
COPY --from=intelanalytics/bigdl-ppml-trusted-bigdata-gramine-reference-8g:2.3.0-SNAPSHOT ${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar /ppml/jars
ENTRYPOINT [ "/ppml/bigdl-kms-entrypoint.sh" ]

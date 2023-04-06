ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG SPARK_VERSION=3.1.3

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

FROM keinos/sqlite3 as sqlite
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /bigdl-${BIGDL_VERSION}
ARG http_proxy
ARG https_proxy
ADD ./init.sh /ppml/init.sh
RUN mkdir /ppml && \
    cd /ppml && \
    wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-ppml-spark_${SPARK_VERSION}/${BIGDL_VERSION}/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar && \
    chmod a+x /ppml/init.sh
COPY --from=java /opt/jdk  /opt/jdk8
ENTRYPOINT [ "/ppml/init.sh" ]

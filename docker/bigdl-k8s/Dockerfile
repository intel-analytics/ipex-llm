ARG SPARK_VERSION=2.4.6
ARG SPARK_HOME=/opt/spark
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0

# stage.1 jdk & spark
FROM ubuntu:18.04 as spark
ARG SPARK_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_HOME
ARG TINI_VERSION
ENV TINI_VERSION                        ${TINI_VERSION}
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          ${SPARK_HOME}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
# java
    wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk && \
# spark
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    cp /opt/spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt

RUN ln -fs /bin/bash /bin/sh
RUN if [ $SPARK_VERSION = "3.1.2" ]; then \
        rm $SPARK_HOME/jars/okhttp-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar; \
    elif [ $SPARK_VERSION = "2.4.6" ]; then \
        rm $SPARK_HOME/jars/kubernetes-client-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/io/fabric8/kubernetes-client/4.4.2/kubernetes-client-4.4.2.jar; \
    fi

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini

# stage.2 bigdl
FROM ubuntu:18.04 as bigdl
ARG SPARK_VERSION
ARG BIGDL_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /opt/bigdl-${BIGDL_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git
ADD ./download-bigdl.sh /opt

RUN chmod a+x /opt/download-bigdl.sh && \
    mkdir -p /opt/bigdl-examples/python
RUN /opt/download-bigdl.sh && \
    rm bigdl*.zip

# stage.3 copies layer
FROM ubuntu:18.04 as copies-layer
ARG BIGDL_VERSION

COPY --from=bigdl /opt/bigdl-${BIGDL_VERSION} /opt/bigdl-${BIGDL_VERSION}
COPY --from=spark /opt/jdk /opt/jdk
COPY --from=spark /opt/spark /opt/spark
COPY --from=spark /opt/spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt


# stage.4
FROM ubuntu:18.04
MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo
ARG BIGDL_VERSION
ARG SPARK_VERSION
ARG SPARK_HOME
ARG TINI_VERSION

ENV BIGDL_HOME                          /opt/bigdl-${BIGDL_VERSION}
ENV BIGDL_VERSION                       ${BIGDL_VERSION}
ENV SPARK_HOME                          ${SPARK_HOME}
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV OMP_NUM_THREADS                     4
ENV NOTEBOOK_PORT                       12345
ENV NOTEBOOK_TOKEN                      1234qwer
ENV RUNTIME_SPARK_MASTER                local[4]
ENV RUNTIME_K8S_SERVICE_ACCOUNT         spark
ENV RUNTIME_K8S_SPARK_IMAGE             intelanalytics/bigdl-k8s:${BIGDL_VERSION}-${SPARK_VERSION}
ENV RUNTIME_DRIVER_HOST                 localhost
ENV RUNTIME_DRIVER_PORT                 54321
ENV RUNTIME_EXECUTOR_CORES              4
ENV RUNTIME_EXECUTOR_MEMORY             20g
ENV RUNTIME_EXECUTOR_INSTANCES          1
ENV RUNTIME_TOTAL_EXECUTOR_CORES        4
ENV RUNTIME_DRIVER_CORES                4
ENV RUNTIME_DRIVER_MEMORY               10g
ENV RUNTIME_PERSISTENT_VOLUME_CLAIM     myvolumeclaim
ENV SPARK_HOME                          /opt/spark
ENV BIGDL_CLASSPATH                     ${BIGDL_HOME}/jars/*.jar
ENV JAVA_HOME                           /opt/jdk
ENV PYTHONPATH                          ${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip:${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip:${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip:${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip:${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-*.zip:/opt/models/research/slim
ENV TINI_VERSION                        ${TINI_VERSION}
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8


COPY --from=copies-layer /opt /opt
COPY --from=spark /sbin/tini /sbin/tini

RUN mkdir -p /opt/bigdl-examples/python && \
    mkdir -p /opt/bigdl-examples/scala && \
    apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
    apt-get install -y gcc g++ make && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 
# python
ADD ./install-conda-env.sh /opt
RUN chmod a+x /opt/install-conda-env.sh 
RUN /opt/install-conda-env.sh && \
# chmod
RUN chmod +x /sbin/tini && \
    cp /sbin/tini /usr/bin/tini

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

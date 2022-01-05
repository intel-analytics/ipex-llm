ARG SPARK_VERSION=2.4.6
ARG HADOOP_VERSION=2.7
ARG SPARK_HOME=/opt/spark
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ARG BIGDL_VERSION=0.14.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG PYTHON_ENV_NAME=tf1

# stage.1 jdk & spark
FROM ubuntu:18.04 as spark
ARG SPARK_VERSION
ARG HADOOP_VERSION
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
    wget -O jdk.tar.gz $JDK_URL && \
    gunzip jdk.tar.gz && \
    mkdir /opt/jdk$JDK_VERSION && \
    tar -xf jdk.tar -C /opt/jdk$JDK_VERSION --strip-components 1 && \
    rm jdk.tar && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk && \
# spark
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
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
RUN git clone https://github.com/tensorflow/models/ /opt/models

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
ARG PYTHON_ENV_NAME

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
ENV JAVA_HOME                           /opt/jdk
ENV BIGDL_CLASSPATH                     ${BIGDL_HOME}/jars/*
ENV PYTHON_ENV_NAME                     ${PYTHON_ENV_NAME}
ENV PYTHONPATH                          ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip:${BIGDL_HOME}/conf/spark-bigdl.conf:${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-*.zip:/opt/models/research/slim
ENV PATH ${BIGDL_HOME}/bin/cluster-serving:${JAVA_HOME}/bin:${PATH}
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
# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    chmod +x Miniconda3-4.5.4-Linux-x86_64.sh && \
    ./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local 

ADD ./install-python-env1.sh /opt
ADD ./install-python-env2.sh /opt
RUN chmod a+x /opt/install-python-env1.sh && \
    chmod a+x /opt/install-python-env2.sh && \
    chmod +x /sbin/tini
RUN if [ $PYTHON_ENV_NAME = "tf1" ]; then \
        rm /opt/install-python-env2.sh && \
        /opt/install-python-env1.sh; \
    elif [ $PYTHON_ENV_NAME = "tf2" ]; then \
        rm /opt/install-python-env1.sh && \
        /opt/install-python-env2.sh; \
    fi
RUN cp /sbin/tini /usr/bin/tini

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]
